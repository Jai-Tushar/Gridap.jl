using Gridap
using Gridap.ReferenceFEs, Gridap.FESpaces, Gridap.Fields
using DrWatson


function variational(domain, cells, order, yex, pex, zex)

  f(x) = -Δ(yex)(x) - zex(x)
  yd(x) = yex(x) + Δ(pex)(x)

  model = simplexify(CartesianDiscreteModel(domain, cells))
  reffe = ReferenceFE(lagrangian, Float64, order; space=:P)
  Vs = FESpace(model, reffe; conformity=:H1, dirichlet_tags="boundary")
  Us = TrialFESpace(Vs, yex)
  Va = Vs
  Ua = TrialFESpace(Va, pex)

  X = MultiFieldFESpace([Us, Ua])
  Y = MultiFieldFESpace([Vs, Va])

  Ω  = Triangulation(model)
  dΩ = Measure(Ω, 2*(order+1))
  dΩz = Measure(Ω, 10*(order+1))

  function _max(a,b)
    return (a + b + abs(a - b)) / 2
  end

  function _min(a,b)
    return (a + b - abs(a - b)) / 2
  end

  _Proj(p) = _max(za, _min(zb, (-1/λ) * p))

  res( (y,p),(v,s) ) = ∫(∇(y)⋅∇(v))dΩ - ∫(f*v)dΩ - ∫((_Proj∘p)*v)dΩ + ∫(∇(p)⋅∇(s))dΩ - ∫(y*s)dΩ + ∫(yd*s)dΩ
  # res( (y,p),(v,s) ) = ∫(∇(y)⋅∇(v))dΩ - ∫(f*v)dΩ - ∫(_Proj(p)*v)dΩ + ∫(∇(p)⋅∇(s))dΩ - ∫(y*s)dΩ + ∫(yd*s)dΩ

  # Initial guess of free dofs
  wh = FEFunction(X, zeros(num_free_dofs(X)))

  dv = get_fe_basis(Y)
  res_jac(wh) = res(wh, dv)

  assem = SparseMatrixAssembler(X,Y)

  iter = 0
  Er = 1.0
  tol = 1e-8
  max_iter = 10

  while Er > tol && iter < max_iter
    iter += 1
    J = jacobian(res_jac, wh)
    r = res_jac(wh)
    data = collect_cell_matrix_and_vector(X,Y,J,r,zero(X))
    A,b = assemble_matrix_and_vector(assem,data)

    # x_{n+1} = x_n - J^{-1} * r, where δsol = J^{-1} * r
    δwh = A \ b
    δwh = FEFunction(X, δwh)
    wh = FEFunction(X, wh.free_values - δwh.free_values)
    δyh, δph = δwh
    Er = sqrt(sum(∫((δyh * δyh) + (δph * δph))*dΩ))

    # wh = wh.free_values - δwh
    # wh = FEFunction(X, wh)
    # Er = norm(δwh)
    @info "Iteration: $iter, Residual: $Er"
  end


  # optimal solution
  yh, ph = wh
  zh = _Proj∘(ph)
  # zh = _Proj(ph)

  ey = yh - yex
  ep = ph - pex
  ez = zh - zex

  el2_yh = sqrt(sum(∫(ey⋅ey)*dΩ))
  el2_ph = sqrt(sum(∫(ep⋅ep)*dΩ))

  el2_zh = sqrt(sum(∫(ez⋅ez)*dΩz))

  dofs = num_free_dofs(X)

  return el2_yh, el2_ph, el2_zh, dofs, zh

end

function slope(hs,errors)
  x = log10.(hs)
  y = log10.(errors)
  linreg = hcat(fill!(similar(x), 1), x) \ y
  linreg[2]
end

function convg_test(domain,ncs,order,yex, pex, zex)
  el2y = Float64[]
  el2p = Float64[]
  el2z = Float64[]
  hs = Float64[]
  ndofs = Int[]
  for nc in ncs
    println(repeat("=", 50))
    println("nc = $nc") 
    el2_yh, el2_ph, el2_zh, dof = variational(domain, nc, order, yex, pex, zex)
    push!(el2y, el2_yh)
    push!(el2p, el2_ph)
    push!(el2z, el2_zh)
    push!(ndofs, dof)
    h = 1/nc[1]
    push!(hs, h)
  end
  return el2y, el2p, el2z, hs, ndofs
end


za = -5 
zb = 5 
λ = 0.001

yex(x) = sin(π*x[1])*sin(π*x[2])
pex(x) = -2*pi^2*sin(π*x[1])*sin(π*x[2])
zex(x) = max( za, min( zb, -(1/λ) * pex(x) ) )

order  = 2
domain = (0,1,0,1)
ncs = [(2,2),(4,4),(8,8),(16,16),(32,32),(64,64),(128,128)]

el2ys, el2ps, el2zs, hs, ndofs = convg_test(domain, ncs, order, yex, pex, zex);


println("L2-error yΩ: ", el2ys)
println("Slope L2-norm state: $(2*slope(ndofs,el2ys))")
println("Slope L2-norm state: $(slope(hs,el2ys))")

println("L2-error pΩ: ", el2ps)
println("Slope L2-norm adjoint: $(2*slope(ndofs,el2ps))")
println("Slope L2-norm state: $(slope(hs,el2ps))")

println("L2-error zΩ: ", el2zs)
println("Slope L2-norm control: $(2*slope(ndofs,el2zs))")
println("Slope L2-norm state: $(slope(hs,el2zs))")


el2_y, el2_p, el2_z, dofs, opt_z = variational(domain, (128,128), order, yex, pex, zex)
Ω = get_triangulation(opt_z)
writevtk(Ω, "optimal_control", cellfields=["optimal_control" => opt_z];append=false)