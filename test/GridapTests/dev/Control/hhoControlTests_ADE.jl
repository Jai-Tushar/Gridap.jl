using Gridap
using Gridap.Geometry, Gridap.FESpaces, Gridap.MultiField
using Gridap.CellData, Gridap.Fields, Gridap.Helpers
using Gridap.ReferenceFEs
using Gridap.Arrays

function projection_operator(V, Ω, dΩ)
  Π(u,Ω) = change_domain(u,Ω,DomainStyle(u))
  mass(u,v) = ∫(u⋅Π(v,Ω))dΩ
  V0 = FESpaces.FESpaceWithoutBCs(V)
  P = FESpaces.LocalOperator(
    FESpaces.LocalSolveMap(), V0, mass, mass; trian_out = Ω
  )
  return P
end

function reconstruction_operator(ptopo,L,X,Ω,Γp,dΩp,dΓp)
  reffe_Λ = ReferenceFE(lagrangian, Float64, 0; space=:P)
  Λ = FESpace(Ω, reffe_Λ; conformity=:L2)

  nrel = get_normal_vector(Γp)
  Πn(v) = ∇(v)⋅nrel
  Π(u,Ω) = change_domain(u,Ω,DomainStyle(u))
  lhs((u,λ),(v,μ))   =  ∫( (∇(u)⋅∇(v)) + (μ*u) + (λ*v) )dΩp
  rhs((uT,uF),(v,μ)) =  ∫( (∇(uT)⋅∇(v)) + (uT*μ) )dΩp + ∫( (uF - Π(uT,Γp))*(Πn(v)) )dΓp
  
  Y = FESpaces.FESpaceWithoutBCs(X)
  mfs = Y.multi_field_style
  W = MultiFieldFESpace([L,Λ];style=mfs)
  R = LocalOperator(
    LocalPenaltySolveMap(), ptopo, W, Y, lhs, rhs; space_out = L
  )
  return R
end

function as(u,v)
    Ru_Ω, Ru_Γ = Rs(u)
    Rv_Ω, Rv_Γ = Rs(v)
    return ∫(∇(Ru_Ω)⋅∇(Rv_Ω) + ∇(Ru_Γ)⋅∇(Rv_Ω) + ∇(Ru_Ω)⋅∇(Rv_Γ) + ∇(Ru_Γ)⋅∇(Rv_Γ))dΩp
end 

function ss(u,v)
    function SΓs(u)
        u_Ω, u_Γ = u
        return PΓs(u_Ω) - u_Γ
    end
    return ∫(hTinv * (SΓs(u)⋅SΓs(v)))dΓp
end

yex(x) = sin(2π*x[1])*sin(2π*x[2])


order  = 1
domain = (0,1,0,1)

cells = (2,2)
model = UnstructuredDiscreteModel(CartesianDiscreteModel(domain,cells))
D = num_cell_dims(model)
Ω = Triangulation(ReferenceFE{D}, model)
Γ = Triangulation(ReferenceFE{D-1}, model)

ptopo = Geometry.PatchTopology(model)
Ωp = Geometry.PatchTriangulation(model,ptopo)
Γp = Geometry.PatchBoundaryTriangulation(model,ptopo)

qdegree = 2*(order+1)

dΩp = Measure(Ωp,qdegree)
dΓp = Measure(Γp,qdegree)


reffe_V = ReferenceFE(lagrangian, Float64, order+1; space=:P)   # Bulk space
reffe_M = ReferenceFE(lagrangian, Float64, order; space=:P)     # Skeleton space
reffe_L = ReferenceFE(lagrangian, Float64, order+1; space=:P)   # Reconstruction space


Vs = FESpace(Ω, reffe_V; conformity=:L2)
Ms = FESpace(Γ, reffe_M; conformity=:L2, dirichlet_tags="boundary")
Ls = FESpace(Ω, reffe_L; conformity=:L2)
Ns = TrialFESpace(Ms,yex)

mfs = MultiField.BlockMultiFieldStyle(2,(1,1))
Xs = MultiFieldFESpace([Vs, Ns];style=mfs)
Ys = MultiFieldFESpace([Vs, Ms];style=mfs)

PΓs = projection_operator(Ms, Γp, dΓp)
Rs  = reconstruction_operator(ptopo,Ls,Ys,Ωp,Γp,dΩp,dΓp)

T = maximum(sqrt(2).*sqrt.(get_array(∫(1)dΩp)))
hTinv =  CellField(1 ./ (sqrt(2).*sqrt.(get_array(∫(1)dΩp))),Ωp)

function as(u,v)
    Ru_Ω, Ru_Γ = Rs(u)
    Rv_Ω, Rv_Γ = Rs(v)
#     return ∫(∇(Ru_Ω)⋅∇(Rv_Ω) + ∇(Ru_Γ)⋅∇(Rv_Ω) + ∇(Ru_Ω)⋅∇(Rv_Γ) + ∇(Ru_Γ)⋅∇(Rv_Γ))dΩp
    return ∫(∇(Ru_Ω + Ru_Γ) ⋅ ∇(Rv_Ω + Rv_Γ))dΩp    
end

function as(wh::MultiFieldFEFunction,v)
    Rwh = Rs(wh)
    Rv_Ω, Rv_Γ = Rs(v)
    return ∫(∇(Rwh)⋅∇(Rv_Ω + Rv_Γ))dΩp
end
    
function ms(u,v)
    uΩ, uΓ = u 
    vΩ, vΓ = v
    return ∫(uΩ*vΩ)dΩp
end


u, v = get_trial_fe_basis(Xs), get_fe_basis(Ys)
 


Xp = FESpaces.PatchFESpace(Xs,ptopo)
wh = FEFunction(Xp, zeros(num_free_dofs(Xp)))
dv = get_fe_basis(Ys)

res_m(u,v) = ms(u,v)
resm_jac(wh) = res_m(wh, dv)
rm = resm_jac(wh)
Jm = jacobian(resm_jac, wh)
 
res_a(u,v) = as(u,v)
resa_jac(wh) = res_a(wh, dv)
ra = resa_jac(wh)
@enter Ja = jacobian(resa_jac, wh; ad_type=:monolithic)

fuh = resa_jac(wh)



du = get_trial_fe_basis(Xs)

ah = as(du,dv)




















# try
#     Ja = jacobian(resa_jac, wh)
# catch e
#     bt = stacktrace(catch_backtrace())
#     for (i,f) in enumerate(bt)
#         println(i, " ", f)
#     end
#     rethrow()
# end

# Rs(u)
# Rwh = Rs(wh)
# RdvΩ, RdvΓ = Rs(dv)

# ∫(∇(Rwh)⋅∇(RdvΩ + RdvΓ))dΩp
