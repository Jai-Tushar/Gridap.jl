using Gridap
using Gridap.Geometry, Gridap.FESpaces, Gridap.MultiField
using Gridap.TensorValues: outer
using Gridap.CellData

uex(x) = sin(π*x[1])^2 * sin(π*x[2])^2
domain = (0,1,0,1) 
nc = (3,3)

model = UnstructuredDiscreteModel(CartesianDiscreteModel(domain,nc))
model = Geometry.PolytopalDiscreteModel(model)
D = num_cell_dims(model)
Ω = Triangulation(ReferenceFE{D}, model)
Γ = Triangulation(ReferenceFE{D-1}, model)

ptopo = Geometry.PatchTopology(model)
Ωp = Geometry.PatchTriangulation(model,ptopo)
Γp = Geometry.PatchBoundaryTriangulation(model,ptopo)

nrel = get_normal_vector(Γp) 

I = one(TensorValue{D,D,Float64})

# Tangential projector
tP(n) = I - outer(n,n)

∂n(v,n)= ∇(v)⋅n

∂t(v,n)= tP(n) ⋅ ∇(v)

H(v) = ∇(∇(v))   # ( (∇⊗∇)(v) )
# ∂nn(v,n) = n' * H(v)*n
∂nn(v,n) = n ⋅ (H(v) ⋅ n)

∂nt(v,n) = tP(n) ⋅ (H(v) ⋅ n)

order = 0
reffe_M = ReferenceFE(lagrangian, Float64, order+1)     # Trace approximation space
reffe_L = ReferenceFE(lagrangian, Float64, order)       # Normal derivative approximation space
V = FESpaces.PolytopalFESpace(Ω, Float64, order+2; space=:P)        # Bulk approximation space
M = FESpace(Γ, reffe_M; conformity=:L2, dirichlet_tags="boundary")
L = FESpace(Γ, reffe_L; conformity=:L2)
N = TrialFESpace(M,uex)

mfs = MultiField.BlockMultiFieldStyle(2,(1,2))
X = MultiFieldFESpace([V, N, L]; style=mfs) # Trial space
Y = MultiFieldFESpace([V, M, L]; style=mfs) # Test space
Xp = FESpaces.PatchFESpace(X, ptopo) 

u = get_trial_fe_basis(X)
v = get_fe_basis(Y)

uΩ, uΓ, uΓn = u
vΩ, vΓ, vΓn = v

# Checking operations
H(vΩ)
∂n(Δ(vΩ),nrel)
∂nn(vΩ,nrel)
∂nt(vΩ,nrel)
Δ(Δ(vΩ))

Π(u,Ω) = change_domain(u,Ω,DomainStyle(u))

qdegree = 4*(order+1)
dΩp = Measure(Ωp,qdegree)
dΓp = Measure(Γp,qdegree)

# checking integrals
inner(H(vΩ),H(vΩ))

∫(H(uΩ) ⊙ H(vΩ))dΩp
∫( Π(uΩ,Γp) - uΓ)dΓp

 

∫((∂n(vΩ,nrel)-vΓn))dΓp
∫(∂nn(vΩ,nrel))dΓp
∫(∂nt(vΩ,nrel))dΓp
∫(∂t(vΩ,nrel))dΓp


∫((∂n(vΩ,nrel) - vΓn)*∂nn(uΩ,nrel))dΓp 

∫(∂n(vΩ,nrel)*∂nt(uΩ,nrel))dΓp
∫(∂t(vΓ,nrel) ⋅ ∂nt(uΩ,nrel))dΓp
∫(∂t(Π(vΩ,Γp)-vΓ,nrel) ⋅ ∂nt(uΩ,nrel))dΓp

cf = ∂n(Δ(vΩ),nrel)
∫(cf)dΓp