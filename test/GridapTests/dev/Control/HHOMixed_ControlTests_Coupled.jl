"""
References:

- A Variational Discretization Concept in Control Constrained Optimization: The Linear-Quadratic Case. M. Hinze. 2005

"""

using Gridap
using Gridap.Geometry, Gridap.FESpaces, Gridap.MultiField
using Gridap.CellData, Gridap.Fields, Gridap.Helpers
using Gridap.ReferenceFEs

function l2_error(uh,u,dΩ)
  eh = uh - u
  return sqrt(sum(∫(eh⋅eh)*dΩ))
end

"""
# Input:
- `u`: A `MultiFieldCellField`, `MultiFieldFEFunction`, or similar MultiField object composed of subfields.
- `ids`: A vector of integers specifying the new field ID for each component of `u`.
- `nfields`: Total number of fields in the full global system.

# Output:
A new `MultiFieldCellField` (or similar) with the same data but updated internal field IDs,
allowing the object to be correctly interpreted during block-wise global assembly.
"""
function swap_field_ids(u, ids, nfields)
  @assert length(ids) == length(u)
  fields = map((ui,id) -> swap_field_ids(ui,id,nfields), u, ids)
  return MultiField.MultiFieldCellField(fields)
end

function swap_field_ids(u::MultiField.MultiFieldFEBasisComponent, id, nfields)
  return MultiField.MultiFieldFEBasisComponent(u.single_field, id, nfields)
end

# function swap_field_ids(u::Gridap.FESpaces.SingleFieldFEBasis, id, nfields)
#   return  MultiField.MultiFieldFEBasisComponent(u, id, nfields)
# end

# function projection_operator_adjoint(V, Ω, dΩ)
#   Π(u,Ω) = change_domain(u,Ω,DomainStyle(u))
#   mass(u,v) = ∫(u⋅Π(v,Ω))dΩ
#   V0 = FESpaces.FESpaceWithoutBCs(V)
#   P = FESpaces.LocalOperator(
#     FESpaces.LocalSolveMap(), V0, mass, mass; trian_out = Ω
#   )
#   _P(u) = swap_field_ids(CellData.similar_cell_field(u.single_field,get_data(P(u)), get_triangulation(P(u)), ReferenceDomain()),3,4)
#   return _P
# end

# function projection_operator_adjoint(V, Ω, dΩ)
#   Π(u,Ω) = change_domain(u,Ω,DomainStyle(u))
#   mass(u,v) = ∫(u⋅Π(v,Ω))dΩ
#   V0 = FESpaces.FESpaceWithoutBCs(V)
#   P = FESpaces.LocalOperator(
#     FESpaces.LocalSolveMap(), V0, mass, mass; trian_out = Ω
#   )
#   _P(u) = swap_field_ids(CellData.similar_cell_field(u.single_field,get_data(P(u)), get_triangulation(P(u)), ReferenceDomain()),4,4)
#   return _P
# end

function projection_operator(V, Ω, dΩ)
  Π(u,Ω) = change_domain(u,Ω,DomainStyle(u))
  mass(u,v) = ∫(u⋅Π(v,Ω))dΩ
  V0 = FESpaces.FESpaceWithoutBCs(V)
  P = FESpaces.LocalOperator(
    FESpaces.LocalSolveMap(), V0, mass, mass; trian_out = Ω
  )
  return P
end

function reconstruction_operator_state(ptopo,L,X,Ω,Γp,dΩp,dΓp)
  reffe_Λ = ReferenceFE(lagrangian, Float64, 0; space=:P)
  Λ = FESpace(Ω, reffe_Λ; conformity=:L2)

  nrel = get_normal_vector(Γp)
  Πn(v) = ∇(v)⋅nrel
  Π(u,Ω) = change_domain(u,Ω,DomainStyle(u))
  lhs((u,λ),(v,μ))   =  ∫( (∇(u)⋅∇(v)) + (μ*u) + (λ*v) )dΩp
  rhs((uT,uF),(v,μ)) =  ∫( (∇(uT)⋅∇(v)) )dΩp + ∫( (uT*μ) )dΩp + ∫( (uF - Π(uT,Γp))*(Πn(v)) )dΓp
  
  Y = FESpaces.FESpaceWithoutBCs(X)
  mfs = Y.multi_field_style
  W = MultiFieldFESpace([L,Λ];style=mfs)
  R = LocalOperator(
    LocalPenaltySolveMap(), ptopo, W, Y, lhs, rhs; space_out = L
  )
  _R(u) = swap_field_ids(R(swap_field_ids(u,[1,2],2)),[1,3],4)
  return _R
end

function reconstruction_operator_adjoint(ptopo,L,X,Ω,Γp,dΩp,dΓp)
  reffe_Λ = ReferenceFE(lagrangian, Float64, 0; space=:P)
  Λ = FESpace(Ω, reffe_Λ; conformity=:L2)

  nrel = get_normal_vector(Γp)
  Πn(v) = ∇(v)⋅nrel
  Π(u,Ω) = change_domain(u,Ω,DomainStyle(u))
  lhs((u,λ),(v,μ))   =  ∫( (∇(u)⋅∇(v)) + (μ*u) + (λ*v) )dΩp
  rhs((uT,uF),(v,μ)) =  ∫( (∇(uT)⋅∇(v)) )dΩp + ∫( (uT*μ) )dΩp + ∫( (uF - Π(uT,Γp))*(Πn(v)) )dΓp
  
  Y = FESpaces.FESpaceWithoutBCs(X)
  mfs = Y.multi_field_style
  W = MultiFieldFESpace([L,Λ];style=mfs)
  R = FESpaces.LocalOperator(
    FESpaces.HHO_ReconstructionOperatorMap(), ptopo, W, Y, lhs, rhs; space_out = L
  )
  _R(u) = swap_field_ids(R(swap_field_ids(u,[1,2],2)),[2,4],4)
  return _R
end

##############################################################
yex(x) = 2*pi^2*sin(2π*x[1])*sin(2π*x[2])
pex(x) = sin(pi*x[1])*sin(pi*x[2])
za = -15; zb = 15; λ = 0.25
zex(x) = max( za, min(zb, -(1/λ) * pex(x)) )
f(x) = -Δ(yex)(x) - zex(x)
yd(x) = yex(x) + Δ(pex)(x)

nc = (2,2)
model = UnstructuredDiscreteModel(CartesianDiscreteModel((0,1,0,1),nc))
D = num_cell_dims(model)
Ω = Triangulation(ReferenceFE{D}, model)
Γ = Triangulation(ReferenceFE{D-1}, model)

ptopo = Geometry.PatchTopology(model)
Ωp = Geometry.PatchTriangulation(model,ptopo)
Γp = Geometry.PatchBoundaryTriangulation(model,ptopo)

order = 0
qdegree = 2*(order+1)

dΩp = Measure(Ωp,qdegree)
dΓp = Measure(Γp,qdegree)

##########################
# Mixed order variant
##########################
reffe_V = ReferenceFE(lagrangian, Float64, order+1; space=:P)   # Bulk space
reffe_M = ReferenceFE(lagrangian, Float64, order; space=:P)     # Skeleton space
reffe_L = ReferenceFE(lagrangian, Float64, order+1; space=:P)   # Reconstruction space

# State spaces
Vs = FESpace(Ω, reffe_V; conformity=:L2)
Ms = FESpace(Γ, reffe_M; conformity=:L2, dirichlet_tags="boundary")
Ns = TrialFESpace(Ms,yex)

# Adjoint spaces
Va = FESpace(Ω, reffe_V; conformity=:L2)
Ma = FESpace(Γ, reffe_M; conformity=:L2, dirichlet_tags="boundary")
Na = TrialFESpace(Ma,pex)

# Global spaces
mfs = MultiField.BlockMultiFieldStyle(2,(2,2))
X = MultiFieldFESpace([Vs, Va, Ns, Na];style=mfs)
Y = MultiFieldFESpace([Vs, Va, Ms, Ma];style=mfs)
Xp = FESpaces.PatchFESpace(X,ptopo)

# Reconstruction operator spaces
L = FESpace(Ω, reffe_L; conformity=:L2)
YRs= MultiFieldFESpace([Vs, Ms];style=MultiField.BlockMultiFieldStyle(2,(1,1)))
YRa= MultiFieldFESpace([Va, Ma];style=MultiField.BlockMultiFieldStyle(2,(1,1)))

# Operators
Rs  = reconstruction_operator_state(ptopo,L,YRs,Ωp,Γp,dΩp,dΓp)
Ra  = reconstruction_operator_adjoint(ptopo,L,YRa,Ωp,Γp,dΩp,dΓp)

# PΓs = projection_operator_state(Ms, Γp, dΓp)
# PΓa = projection_operator_adjoint(Ma, Γp, dΓp)

PΓs = projection_operator(Ms, Γp, dΓp)
PΓa = projection_operator(Ma, Γp, dΓp)


patch_assem  = FESpaces.PatchAssembler(ptopo,X,Y)
global_assem = SparseMatrixAssembler(X,Y)

function as(u,v)
  Ru_Ω, Ru_Γ = Rs(u)
  Rv_Ω, Rv_Γ = Rs(v)
  return ∫(∇(Ru_Ω)⋅∇(Rv_Ω) + ∇(Ru_Γ)⋅∇(Rv_Ω) + ∇(Ru_Ω)⋅∇(Rv_Γ) + ∇(Ru_Γ)⋅∇(Rv_Γ))dΩp
end

function aa(u,p,v)
  Rp_Ω, Rp_Γ = Ra(p)
  Rv_Ω, Rv_Γ = Ra(v)
  uΩ, uΓ = u
  vΩ, vΓ = v
  return ∫((∇(Rp_Ω)⋅∇(Rv_Ω) + ∇(Rp_Γ)⋅∇(Rv_Ω) + ∇(Rp_Ω)⋅∇(Rv_Γ) + ∇(Rp_Γ)⋅∇(Rv_Γ)) - (uΩ * vΩ))dΩp
end

hTinv =  CellField(1 ./ (sqrt(2).*sqrt.(get_array(∫(1)dΩp))),Ωp)
function ss(u,v)
  function SΓs(u)
    u_Ω, u_Γ = u
    cs = PΓs(u_Ω)
    cs_basis = CellData.similar_cell_field(uΩ.single_field, get_data(cs), get_triangulation(cs), ReferenceDomain())
    PΓs_uΩ = swap_field_ids(cs_basis, 3, 4)
    return PΓs_uΩ - u_Γ
  end
  return ∫(hTinv * (SΓs(u)⋅SΓs(v)))dΓp
end

function sa(u,v)
  function SΓa(u)
    u_Ω, u_Γ = u
    ca = PΓa(u_Ω)
    ca_basis = CellData.similar_cell_field(uΩ.single_field, get_data(ca), get_triangulation(ca), ReferenceDomain())
    PΓa_uΩ = swap_field_ids(ca_basis, 4, 4)
    return PΓa_uΩ - u_Γ
  end
  return ∫(hTinv * (SΓa(u)⋅SΓa(v)))dΓp
end
 
function _max(a,b)
  return (a+b+abs(a-b))/2
end

function _min(a,b)
  return (a+b-abs(a-b))/2
end

_Proj(p) = _max( za, _min( zb, (-1/λ)*p ) )

ls((vΩ,vΓ),p0) = ∫(f*vΩ + _Proj(p0)*vΩ)dΩp
la((vΩ,vΓ)) = ∫(CellField(-1,Ωp)*yd*vΩ)dΩp 

x, y = get_trial_fe_basis(X), get_fe_basis(Y);

function weakform(X,Y,wh)

  x, y = get_trial_fe_basis(X), get_fe_basis(Y);

  u, p = (x[1],x[3]), (x[2],x[4])
  v, q = (y[1],y[3]), (y[2],y[4])

  y0Ω, p0Ω, y0Γ, p0Γ = wh
  data1 = FESpaces.collect_cell_matrix_and_vector(X,Y,ss(u,v),ls(v,p0Ω),zero(X))
  data2 = FESpaces.collect_cell_matrix_and_vector(Xp,Xp,as(u,v),DomainContribution(),zero(Xp))
  data3 = FESpaces.collect_cell_matrix_and_vector(X,Y,sa(p,q),la(q),zero(X))
  data4 = FESpaces.collect_cell_matrix_and_vector(Xp,Xp,aa(u,p,q),DomainContribution(),zero(Xp))  
  data = FESpaces.merge_assembly_matvec_data(data1,data2,data3,data4)
  return assemble_matrix_and_vector(global_assem,data)
end


function patch_weakform(X,Y,wh)

  x, y = get_trial_fe_basis(X), get_fe_basis(Y);

  u, p = (x[1],x[3]), (x[2],x[4])
  v, q = (y[1],y[3]), (y[2],y[4])

  y0Ω, p0Ω, y0Γ, p0Γ = wh
  data1 = FESpaces.collect_patch_cell_matrix_and_vector(patch_assem,X,Y,ss(u,v),ls(v,p0Ω),zero(X))
  data2 = FESpaces.collect_patch_cell_matrix_and_vector(patch_assem,Xp,Xp,as(u,v),DomainContribution(),zero(Xp))
  data3 = FESpaces.collect_patch_cell_matrix_and_vector(patch_assem,X,Y,sa(p,q),la(q),zero(X))
  data4 = FESpaces.collect_patch_cell_matrix_and_vector(patch_assem,Xp,Xp,aa(u,p,q),DomainContribution(),zero(Xp))  
  data = FESpaces.merge_assembly_matvec_data(data1,data2,data3,data4)
  return assemble_matrix_and_vector(patch_assem,data)
end

function solve_KKT(wh, tol, max_iter)

  iter = 0
  Er = 1.0
  
  while(Er > tol) && (iter < max_iter)
    iter += 1

    op = MultiField.StaticCondensationOperator(X,patch_assem,patch_weakform(X,Y,wh))
    xh = solve(op)
    yΩ, pΩ, yΓ, pΓ = xh
    y0Ω, p0Ω, y0Γ, p0Γ= wh
    Er = norm(get_free_dof_values(pΩ) - get_free_dof_values(p0Ω))
    # Er = norm(get_free_dof_values(xh) - get_free_dof_values(wh))
    println("iter: $iter, Er: $Er")
    wh = FEFunction(X, get_free_dof_values(wh) + get_free_dof_values(xh));
  end
  return xh
end


tol = 1e-8
max_iter = 15

wh = FEFunction(X, zeros(num_free_dofs(X))); # initial guess satisfying boundary conditions
op = MultiField.StaticCondensationOperator(X,patch_assem,patch_weakform(X,Y,wh))
xh = solve(op)

yI, pI, yb, pb = xh;

# Monolithic solve
A, B = weakform(X,Y,wh)
_xh = FEFunction(X, A \ B)

_yI, _pI, _yb, _pb = _xh;

sol = solve_KKT(wh,tol,max_iter)




# ###############

# us, pa = (x[1],x[3]), (x[2],x[4])
# vs, va = (y[1],y[3]), (y[2],y[4])


# y0Ω, p0Ω, y0Γ, p0Γ = wh
# as(us,vs)
# aa(us,pa,va)
# ss(us,vs)
# sa(pa,va)

# Rs(us)


# uΩ, uΓ = us;
# vΩ, vΓ = vs;

# sf_data = get_data(PΓs(uΩ))
# sf = FESpaces.similar_fe_basis(uΩ.single_field,sf_data,PΓs.trian_out,BasisStyle(uΩ),DomainStyle(get_fe_basis(PΓs.space_out)))
# PΓs_uΩ = MultiField.MultiFieldFEBasisComponent(sf,3,4)

# _sf_data = get_data(PΓs(vΩ))
# _sf = FESpaces.similar_fe_basis(vΩ.single_field,_sf_data,PΓs.trian_out,BasisStyle(vΩ),DomainStyle(get_fe_basis(PΓs.space_out)))
# PΓs_vΩ = MultiField.MultiFieldFEBasisComponent(_sf,3,4)

# SΓs_u = PΓs_uΩ - uΓ 
# SΓs_v = PΓs_vΩ - vΓ

# ∫(SΓs_u * SΓs_v)dΓp

# u = uΩ
# nfields, fieldid = u.nfields, u.fieldid
# # block_fields(fields,::TestBasis) = lazy_map(BlockMap(nfields,fieldid),fields)
# # block_fields(fields,::TrialBasis) = lazy_map(BlockMap((1,nfields),fieldid),fields)

# sf = evaluate!(nothing,FESpaces.LocalOperator(),u.single_field)
# data = block_fields(CellData.get_data(sf),BasisStyle(u.single_field))
# return CellData.similar_cell_field(sf,data)


# #################

# uΩ, uΓ = us
# vΩ, vΓ = vs
# pΩ, pΓ = pa
# wΩ, wΓ = va

# function SΓ(u)
#   u_Ω, u_Γ = u
#   # cs = PΓs(u_Ω)
#   # cs_basis = CellData.similar_cell_field(uΩ.single_field, get_data(cs), get_triangulation(cs), ReferenceDomain())
#   # PΓs_uΩ = swap_field_ids(cs_basis, 3, 4)
#   # return PΓs_uΩ - u_Γ
#   return PΓs(u_Ω) - u_Γ
# end

# cf_1 = SΓ(us)*SΓ(pa) 
# ∫(cf_1)dΓp 

# function SΓs(u)
#   u_Ω, u_Γ = u
#   cs = PΓs(u_Ω) 
#   cs_basis = CellData.similar_cell_field(uΓ.single_field, get_data(cs), get_triangulation(cs), ReferenceDomain())
#   PΓs_uΩ = swap_field_ids(cs_basis, 3, 4)
#   return PΓs_uΩ - u_Γ
# end

# cs = PΓs(uΩ) 
# cs_basis = CellData.similar_cell_field(uΓ.single_field, get_data(cs), get_triangulation(cs), ReferenceDomain())
# PΓs_uΩ = swap_field_ids(cs_basis, 3, 4)
# PΓs_uΩ = swap_field_ids(cs, 3, 4)

# ubasis = CellData.similar_cell_field(PΓs(uΩ), get_data(PΓs(uΩ)), get_triangulation(PΓs(uΩ)), ReferenceDomain())

# function swap_field_ids(u::GenericCellField, id, nfields)
#   u_basis = CellData.similar_cell_field(u, get_data(u), get_triangulation(u), ReferenceDomain())
#   return  MultiField.MultiFieldFEBasisComponent(u_basis, id, nfields)
# end

# # ∫(SΓs(us)*SΓs(vs))dΓp
# # PΓa(pΩ)

# # cs = PΓs(uΩ)
# cs_basis = CellData.similar_cell_field(uΩ.single_field, get_data(cs), get_triangulation(cs), ReferenceDomain())
# # PΓs_uΩ = swap_field_ids(cs_basis, 3, 4)

# # ca = PΓa(pΩ)
# # ca_basis = CellData.similar_cell_field(pΩ.single_field, get_data(ca), get_triangulation(ca), ReferenceDomain())
# # PΓa_pΩ = swap_field_ids(ca_basis, 4, 4)

