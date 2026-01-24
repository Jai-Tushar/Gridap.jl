using Gridap
using Gridap.Geometry, Gridap.FESpaces, Gridap.MultiField
using Gridap.TensorValues: outer
using Gridap.CellData
using Gridap.Arrays
using Gridap.ReferenceFEs
using Gridap.CellData, Gridap.Fields, Gridap.Helpers

uex(x) = sin(π*x[1])^2 * sin(π*x[2])^2
domain = (0,1,0,1)
nc = (3,3)

model = UnstructuredDiscreteModel(CartesianDiscreteModel(domain,nc))
model = Geometry.PolytopalDiscreteModel(model)

D  = num_cell_dims(model) 
Ω  = Triangulation(ReferenceFE{D},   model)
Γ  = Triangulation(ReferenceFE{D-1}, model)

ptopo = Geometry.PatchTopology(model)
Ωp = Geometry.PatchTriangulation(model,ptopo)
Γp = Geometry.PatchBoundaryTriangulation(model,ptopo)

# normals
# nK = get_normal_vector(Γp)  
# nF = get_normal_vector(Γ)   # fixed oriented normal per face (global)

I = one(TensorValue{D,D,Float64})

tP(n) = I - outer(n,n)               # tangential projector
∂n(v,n)  = ∇(v) ⋅ n                  # scalar normal derivative
∂t(v,n)  = tP(n) ⋅ ∇(v)              # tangential gradient (vector)
H(v)     = ∇(∇(v))                   # Hessian 
∂nn(v,n) = n ⋅ (H(v) ⋅ n)            
∂nt(v,n) = tP(n) ⋅ (H(v) ⋅ n)        


order = 0

# polynomial degrees: cell k+2, trace k+1, normal-deriv k
reffe_M = ReferenceFE(lagrangian, Float64, order+1) 
reffe_L = ReferenceFE(lagrangian, Float64, order)   

V = FESpaces.PolytopalFESpace(Ω, Float64, order+2; space=:P)  
M = FESpace(Γ, reffe_M; conformity=:L2, dirichlet_tags="boundary")

L = FESpace(Γ, reffe_L; conformity=:L2, dirichlet_tags="boundary")

N  = TrialFESpace(M, uex)
gN(x) = 0.0                      # for this uex, ∂n uex = 0 on boundary anyway
Ln = TrialFESpace(L, gN)

mfs = MultiField.BlockMultiFieldStyle(2,(1,2))  
X  = MultiFieldFESpace([V, N,  Ln]; style=mfs)  # Trial
Y  = MultiFieldFESpace([V, M,  L ]; style=mfs)  # Test
Xp = FESpaces.PatchFESpace(X, ptopo)


qdegree = 2*(order+2) + 2
dΩp = Measure(Ωp, qdegree)
dΓp = Measure(Γp, qdegree)


f(x) = Δ(Δ(uex))(x)


function projection_operator(V, Ω, dΩ)
  Π(u,Ω) = change_domain(u,Ω,DomainStyle(u))
  mass(u,v) = ∫( u * Π(v,Ω) )dΩ
  V0 = FESpaces.FESpaceWithoutBCs(V)
  return LocalOperator(LocalSolveMap(), V0, mass, mass; trian_out = Ω)
end

PM = projection_operator(M, Γp, dΓp) 
PL = projection_operator(L, Γp, dΓp) 



function reconstruction_operator(ptopo, order, X, Ωp, Γp, Γ, dΩp, dΓp)

  Π(u,Ω) = change_domain(u,Ω,DomainStyle(u))

  # Local unknown spaces: r_K ∈ P^{k+2}(K), λ_K ∈ P^1(K)
  Rsp = FESpaces.PolytopalFESpace(Ωp, Float64, order+2; space=:P)
  Λsp = FESpaces.PolytopalFESpace(Ωp, Float64, 1;        space=:P)

  # Cell-boundary normal (outward on each ∂K copy)
  nrel = get_normal_vector(Γp)


  nF  = get_normal_vector(Γ)
  sgn = (Π(nF,Γp) ⋅ nrel)  

  # Local saddle-point: (H r, H w)_K + (λ,w)_K + (μ,r)_K = RHS + (μ,vK)_K
  lhs((r,λ),(w,μ)) = ∫( (H(r) ⊙ H(w)) + (λ*w) + (μ*r) )dΩp

  # This is exactly (3.2) with the k=0 simplification (no ∂nΔw term)
  rhs((vK,vF,γF),(w,μ)) 
    vF_p = Π(vF,Γp)
    γF_p = Π(γF,Γp)
    γ∂K  = sgn * γF_p

    # volume term (∇² vK, ∇² w)_K + constraint (vK, μ)_K
    t_vol = ∫( (H(vK) ⊙ H(w)) + (vK*μ) )dΩp

    # boundary: -(∂n vK - γ∂K, ∂nn w)_∂K
    t_nn  = -∫( (∂n(vK,nK) - γ∂K) * ∂nn(w,nK) )dΓp

    # boundary: -(∂t(vK - v∂K), ∂nt w)_∂K
    # implement ∂t(vK - v∂K) = ∂t(vK) - ∂t(v∂K)
    t_nt  = -∫( (∂t(vK,nK) - ∂t(vF_p,nK)) ⋅ ∂nt(w,nK) )dΓp

    return t_vol + t_nn + t_nt
  end

  # Use spaces without BCs inside the local solver
  X0 = FESpaces.FESpaceWithoutBCs(X)
  W  = MultiFieldFESpace([Rsp,Λsp])

  return LocalOperator(LocalPenaltySolveMap(), ptopo, W, X0, lhs, rhs; space_out=Rsp)
end

# call it like this (you already have ptopo, Ωp, Γp, Γ, dΩp, dΓp, and Y)
R = reconstruction_operator_biharmonic(ptopo, order, Y, Ωp, Γp, Γ, dΩp, dΓp)


R = reconstruction_operator(ptopo, order, Y, Ωp, Γp, Γ, dΩp, dΓp)

# ----------------------------
# hK scalings in stabilization
# ----------------------------
vol = collect(get_array(∫(1)dΩp))  # |K|
hK  = sqrt.(vol)                   # 2D proxy; replace by diameter if you prefer
hKinv  = CellField(1.0 ./ hK, Ωp)
hKinv3 = CellField(1.0 ./ (hK.^3), Ωp)

Π(u,Ω) = change_domain(u,Ω,DomainStyle(u))
sgn = Π(nF,Γp) ⋅ nK

# ----------------------------
# Bilinear forms: a + s
# ----------------------------
function a(u,v)
  RuK, Ru∂, Ruγ = R(u)   # 3 contributions (cell/trace/normal) to RK(û)
  RvK, Rv∂, Rvγ = R(v)
  Ru = RuK + Ru∂ + Ruγ
  Rv = RvK + Rv∂ + Rvγ
  return ∫( H(Ru) ⊙ H(Rv) )dΩp
end

function s(u,v)
  function S0(w)
    wK, w∂, wγ = w
    # h^{-3} * (J_{k+1}(w∂ - wK), J_{k+1}(v∂ - vK))_∂K
    # Here we use an L2 projector onto P_{k+1} on each face (works cleanly in Gridap).
    return PM( Π(w∂,Γp) - Π(wK,Γp) )
  end
  function S1(w)
    wK, w∂, wγ = w
    # h^{-1} * (Π_k(γ_∂K - ∂n wK), Π_k(...))_∂K  with γ_∂K := (nF·nK) γF
    diff = sgn*Π(wγ,Γp) - ∂n(wK, nK)
    return PL(diff)
  end
  return ∫( hKinv3*(S0(u)*S0(v)) + hKinv*(S1(u)*S1(v)) )dΓp
end

l((vK,v∂,vγ)) = ∫( f * vK )dΩp

# ----------------------------
# Assembly (monolithic + patch)
# ----------------------------
global_assem = SparseMatrixAssembler(X,Y)
patch_assem  = FESpaces.PatchAssembler(ptopo,X,Y)

function weakform()
  u, v = get_trial_fe_basis(X), get_fe_basis(Y)
  data = FESpaces.collect_and_merge_cell_matrix_and_vector(
    (Xp, Xp, a(u,v), DomainContribution(), zero(Xp)),
    (X,  Y,  s(u,v), l(v),              zero(X))
  )
  return assemble_matrix_and_vector(global_assem, data)
end

function patch_weakform()
  u, v = get_trial_fe_basis(X), get_fe_basis(Y)
  data = FESpaces.collect_and_merge_cell_matrix_and_vector(patch_assem,
    (Xp, Xp, a(u,v), DomainContribution(), zero(Xp)),
    (X,  Y,  s(u,v), l(v),              zero(X))
  )
  return assemble_matrix_and_vector(patch_assem, data)
end

# ----------------------------
# Monolithic solve
# ----------------------------
A, b = weakform()
x = A \ b
uhK, uh∂, uhγ = FEFunction(X, x)

# reconstructed potential (for post-processing/error checks)
RhK, Rh∂, Rhγ = R((uhK,uh∂,uhγ))
uh_rec = RhK + Rh∂ + Rhγ

e = uh_rec - uex
l2  = sqrt(sum(∫(e*e)dΩp))
h2s = sqrt(sum(∫(H(e) ⊙ H(e))dΩp))
@show l2 h2s

# ----------------------------
# Static condensation solve
# ----------------------------
op = MultiField.StaticCondensationOperator(X, patch_assem, patch_weakform())
uhK, uh∂, uhγ = solve(op)

RhK, Rh∂, Rhγ = R((uhK,uh∂,uhγ))
uh_rec = RhK + Rh∂ + Rhγ

e = uh_rec - uex
l2  = sqrt(sum(∫(e*e)dΩp))
h2s = sqrt(sum(∫(H(e) ⊙ H(e))dΩp))
@show l2 h2s
