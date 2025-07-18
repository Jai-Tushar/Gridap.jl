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
model = Geometry.PolytopalDiscreteModel(model)
D = num_cell_dims(model)
Ω = Triangulation(ReferenceFE{D}, model)
Γ = Triangulation(ReferenceFE{D-1}, model)

ptopo = Geometry.PatchTopology(model)
Ωp = Geometry.PatchTriangulation(model,ptopo)
Γp = Geometry.PatchBoundaryTriangulation(model,ptopo)

order = 0
qdegree = 4*(order+1)

dΩp = Measure(Ωp,qdegree)
dΓp = Measure(Γp,qdegree)

#######################
# Mixed order variant #
#######################
# reffe_V = ReferenceFE(lagrangian, Float64, order+2)     # Bulk approximation space
reffe_M = ReferenceFE(lagrangian, Float64, order+1)     # Trace approximation space
reffe_L = ReferenceFE(lagrangian, Float64, order)       # Normal derivative approximation space
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
∂n(Δ(yΩ),nrel)
H(yΩ)
∂nn(yΩ,nrel)
∂nt(yΩ,nrel)

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

∫()

∫( (_Δ(yΩ)) )dΩp


@enter _cf = ∇Δ(yΩ)
@enter  cf = ∇∇(yΩ) 

∫( ∇Δ(yΩ))dΩp

# ∫( ∂n(_Δ(yΩ),nrel))dΓp




# ∫(∇Δ(yΩ))dΩp


####
# 1. H(v) = ∇∇(v) then ∫(H(v) ⊙ H(u))dΩ fails
# 2. cf = ∇ ⋅ ∇(yΩ); ∫( cf )dΩp works but cf = Δ(yΩ); ∫( cf )dΩp fails


#######################################################
∇Δ(f) = gradient(gradient⋅(gradient(f)))

gradient(f,::Val{3}) = ∇Δ(f)

evaluate!(cache,::Broadcasting{typeof(∇Δ)},a::Field) = ∇Δ(a)
lazy_map(::Broadcasting{typeof(∇Δ)},a::AbstractArray{<:Field}) = lazy_map(∇Δ,a)

function push_∇Δ(a::Field,ϕ::Field)
  @notimplemented """\n
  Third order derivatives of quantities defined in the reference domain not implemented yet.

  This is a feature that we want to have at some point in Gridap.
  If you are ready to help with this implementation, please contact the
  Gridap administrators.
  """
end

########################### Check if this is correct
# Define for ∇Δ 
for op in (:+,:-)
  @eval begin
    function ∇Δ(a::OperationField{typeof($op)})
      f = a.fields
      g = map( ∇Δ, f)
      $op(g...)
    end
  end
end
###########################

######### Do we need this for ∇Δ
# function product_rule_hessian(fun,f1,f2,∇f1,∇f2,∇∇f1,∇∇f2)
#   msg = "Product rule not implemented for product $fun between types $(typeof(f1)) and $(typeof(f2))"
#   @notimplemented msg
# end

# function product_rule_hessian(::typeof(*),f1::Real,f2::Real,∇f1,∇f2,∇∇f1,∇∇f2)
#   ∇∇f1*f2 + ∇∇f2*f1 + ∇f1⊗∇f2 + ∇f2⊗∇f1
# end

# for op in (:*,)
#   @eval begin
#     function ∇∇(a::OperationField{typeof($op)})
#       f = a.fields
#       @notimplementedif length(f) != 2
#       f1, f2 = f
#       g1, g2 = map(gradient, f)
#       h1, h2 = map(∇∇, f)
#       prod_rule_hess(F1,F2,G1,G2,H1,H2) = product_rule_hessian($op,F1,F2,G1,G2,H1,H2)
#       Operation(prod_rule_hess)(f1,f2,g1,g2,h1,h2)
#     end
#   end
# end
###########################################

function Fields.evaluate!(cache,k::Broadcasting{typeof(∇Δ)},a::VoidBasis)
  VoidBasis(k(a.basis),a.isvoid)
end


#############################################
# FieldArrays.jl

function return_value(k::Broadcasting{typeof(∇Δ)},a::AbstractArray{<:Field})
  evaluate(k,a)
end

function evaluate!(cache,k::Broadcasting{typeof(∇Δ)},a::AbstractArray{<:Field})
  FieldGradientArray{3}(a)
end

function ∇Δ(a::AbstractArray{<:Field})
  msg =
  """\n
  Double gradient application (aka ∇Δ) is not defined for arrays of Field objects.
  Use Broadcasting(∇Δ) instead.
  """
  @unreachable msg
end

for op in (:∇,:∇∇,:∇Δ)
  @eval begin
    function $op(a::LinearCombinationField)
      fields = Broadcasting($op)(a.fields)
      LinearCombinationField(a.values,fields,a.column)
    end
  end
end

for op in (:∇,:∇∇,:∇Δ)
  @eval begin
    function evaluate!(cache,k::Broadcasting{typeof($op)},a::LinearCombinationFieldVector)
      fields = Broadcasting($op)(a.fields)
      LinearCombinationFieldVector(a.values,fields)
    end
  end
end

evaluate!(cache,k::Broadcasting{typeof(∇Δ)},a::Transpose{<:Field}) = transpose(k(a.parent))

########################################################################
# MacroFEs.jl
function Arrays.evaluate!(cache,k::Broadcasting{typeof(Fields.∇Δ)},a::MacroFEBasis)
  rrule  = a.rrule
  cell_maps = get_cell_map(rrule)
  cell_grads = lazy_map(k,a.fine_data)
  fields = lazy_map(push_∇Δ,cell_grads,cell_maps)
  return FineToCoarseArray(a.rrule,fields,a.ids)
end

########################################################################
# CellFields.jl
function ∇Δ(a::CellField)
  cell_∇Δa = lazy_map(Broadcasting(∇Δ),get_data(a))
  if DomainStyle(a) == PhysicalDomain()
    h = cell_∇Δa
  else
    cell_map = get_cell_map(get_triangulation(a))
    h = lazy_map(Broadcasting(push_∇Δ),cell_∇Δa,cell_map)
  end
  similar_cell_field(a,h,get_triangulation(a),DomainStyle(a))
end

function ∇Δ(a::SkeletonCellFieldPair)
  ∇Δ_cf_plus_plus = ∇Δ(getfield(a,:cf_plus))
  ∇Δ_cf_minus_minus = ∇Δ(getfield(a,:cf_minus))
  SkeletonCellFieldPair(∇Δ_cf_plus_plus,∇Δ_cf_minus_minus)
end

################################################################
# AffineMaps.jl
function push_∇Δ(∇Δa::Field,ϕ::AffineField)
  # Assuming ϕ is affine map
  @notimplemented
end

################################################################
# ApplyOptimizations.jl
function lazy_map(
  k::Broadcasting{typeof(∇Δ)}, a::LazyArray{<:Fill{typeof(linear_combination)}})

  i_to_basis = lazy_map(k,a.args[2])
  i_to_values = a.args[1]
  lazy_map(linear_combination,i_to_values,i_to_basis)
end

function lazy_map(
  k::Broadcasting{typeof(∇Δ)}, a::LazyArray{<:Fill{typeof(transpose)}})

  i_to_basis = lazy_map(k,a.args[1])
  lazy_map(transpose, i_to_basis)
end

#####################################################################
# Autodiff.jl
function ∇Δ(f::Number)
  function _f(x::Point)
    g = gradient(f)(x)
    gradient(g)(x)
  end
end