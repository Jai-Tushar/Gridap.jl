using Gridap
using Gridap.ReferenceFEs, Gridap.FESpaces, Gridap.Fields

cells = (2,2)

# function variational(domain, cells, order, yex, pex, zex)
  f(x) = -Δ(yex)(x) - zex(x)
  yd(x) = yex(x) + Δ(pex)(x)

  model = simplexify(CartesianDiscreteModel(domain, cells))
  reffe = ReferenceFE(lagrangian, Float64, order; space=:P)
  Vs = FESpace(model, reffe; conformity=:H1, dirichlet_tags="boundary")
  Us = TrialFESpace(Vs, yex)
  Va = Vs
  Ua = TrialFESpace(Va, pex)  

  Ω  = Triangulation(model)
  dΩ = Measure(Ω, 2*(order+1))
  
  function _max(a,b)
    return (a + b + abs(a - b)) / 2
  end

  function _min(a,b)
    return (a + b - abs(a - b)) / 2
  end

  _Proj(p) = _max(za, _min(zb, (-1/λ) * p))

  a(u,v) = ∫(∇(u)⋅∇(v))dΩ 
  ls(v,p0)= ∫(f*v)dΩ + ∫(_Proj(p0)*v)dΩ
  la(v,y0)= ∫((y0-yd)*v)dΩ

  p0 = FEFunction(Ua, zeros(num_free_dofs(Ua)))
  hT = maximum(sqrt(2).*sqrt.(get_array(∫(1)dΩ)))

  assems = SparseMatrixAssembler(Us,Vs)
  assema = SparseMatrixAssembler(Ua,Va)

  iter = 0
  Er = 1.0
  tol = 1e-8
  max_iter = 10

#   while Er > tol && iter < max_iter
#     iter += 1

    # state
    v, y = get_fe_basis(Vs), get_trial_fe_basis(Us)
    op = 

    # x_{