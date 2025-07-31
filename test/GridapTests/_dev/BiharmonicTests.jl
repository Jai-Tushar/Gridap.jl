"""
References:
- Hybrid High-Order and Weak Galerkin Methods for the Biharmonic Problem, https://doi.org/10.1137/21M1408555
"""

using Gridap
using Gridap.Geometry, Gridap.FESpaces, Gridap.MultiField
using Gridap.CellData, Gridap.Fields, Gridap.Helpers
using Gridap.ReferenceFEs

function l2_error(uh,u,dΩ)
  eh = uh - u
  return sqrt(sum(∫(eh⋅eh)*dΩ))
end

# normal derivative
∂n(v,n) = ∇(v)⋅n

# tangential derivative
∂t(v,n) = ∇(v) - ∂n(v,n)*n

# second order normal-normal derivative
H(v) = ∇(∇(v))   # ( (∇⊗∇)(v) ) 
∂nn(v,n) = n'*H(v)*n

# normal-tangential derivative
∂nt(v,n) = H(v) - ∂nn(v,n)*n



u(x) = sin(π*x[1])^2 * sin(π*x[2])^2
domain = (0,1,0,1) 
nc = (3,3)


model = UnstructuredDiscreteModel(CartesianDiscreteModel(domain,nc))
# model = Geometry.PolytopalDiscreteModel(model)
D = num_cell_dims(model)
Ω = Triangulation(ReferenceFE{D}, model)
Γ = Triangulation(ReferenceFE{D-1}, model)

ptopo = Geometry.PatchTopology(model)
Ωp = Geometry.PatchTriangulation(model,ptopo)
Γp = Geometry.PatchBoundaryTriangulation(model,ptopo)

order = 0
qdegree = 4*(order+1)

dΩ = Measure(Ω,qdegree)
dΩp = Measure(Ωp,qdegree)
dΓp = Measure(Γp,qdegree)

#######################
# Mixed order variant #
#######################
reffe_V = ReferenceFE(lagrangian, Float64, order+2)     # Bulk approximation space
reffe_M = ReferenceFE(lagrangian, Float64, order+1)     # Trace approximation space
reffe_L = ReferenceFE(lagrangian, Float64, order)       # Normal derivative approximation space

_V = FESpace(Ω, reffe_V; conformity=:L2)
V = FESpaces.PolytopalFESpace(Ω, Float64, order+2; space=:P)        # Bulk approximation space
M = FESpace(Γ, reffe_M; conformity=:L2, dirichlet_tags="boundary")
L = FESpace(Γ, reffe_L; conformity=:L2)
N = TrialFESpace(M,u)

mfs = MultiField.BlockMultiFieldStyle(2,(1,2))
X = MultiFieldFESpace([V, N, L]; style=mfs) # Trial space
Y = MultiFieldFESpace([V, M, L]; style=mfs) # Test space
Xp = FESpaces.PatchFESpace(X, ptopo) 

x = get_trial_fe_basis(X)
y = get_fe_basis(Y)

xΩ, xΓ, xΓ = x
yΩ, yΓ, γΓ = y

nrel = get_normal_vector(Γp)
cf = ∂n(Δ(yΩ),nrel)
H(yΩ)
∂nn(yΩ,nrel)
∂nt(yΩ,nrel)




yy = get_fe_basis(_V)
@enter cf_1 = ∇(Δ(yy))
evaluate(cf_1,cp)
evaluate(yy,cp)


Π(u,Ω) = change_domain(u,Ω,DomainStyle(u))

∫( H(xΩ) ⊙ H(yΩ) )dΩp
∫( Π(xΩ,Γp) )dΓp

# ∫((∂n(Δ(yΩ),nrel)))dΓp  ##



function reconstruction_operator(ptopo,L,X,Ω,Γp,dΩp,dΓp)

  Λ = FESpaces.PolytopalFESpace(Ω, Float64, 1; space=:P)

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


∫(_Δ(yΩ)*_Δ(xΩ))dΩp
∂n(_Δ(yΩ),nrel)


q = get_data(dΩ.quad)

T = get_triangulation(model)
Q = CellQuadrature(T,4)
Q_cd = get_data(Q)
Q_cp = get_cell_points(Q_cd)