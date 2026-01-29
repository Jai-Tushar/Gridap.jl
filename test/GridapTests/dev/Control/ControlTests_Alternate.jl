using Gridap
using Gridap.ReferenceFEs, Gridap.FESpaces, Gridap.Fields

function variational(domain, cells, order, yex, pex, zex;
                     tol=1e-8, max_iter=50)

  # Data (manufactured)
  f(x)  = -Δ(yex)(x) - zex(x)
  yd(x) =  yex(x) + Δ(pex)(x)

  # Model + spaces
  model = simplexify(CartesianDiscreteModel(domain, cells))
  reffe = ReferenceFE(lagrangian, Float64, order; space=:P)

  Vs = FESpace(model, reffe; conformity=:H1, dirichlet_tags="boundary")
  Us = TrialFESpace(Vs, yex)

  Va = Vs
  Ua = TrialFESpace(Va, pex)

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


  a_state(y,v) = ∫( ∇(y)⋅∇(v) )dΩ
  a_adj(p,s)   = ∫( ∇(p)⋅∇(s) )dΩ

  yh = FEFunction(Us, zeros(num_free_dofs(Us)))
  ph = FEFunction(Ua, zeros(num_free_dofs(Ua)))

  iter = 0
  Er = 1.0

  while Er > tol && iter < max_iter
    iter += 1

    yh_old = yh
    ph_old = ph

    # 1) update control from current adjoint
    zh = _Proj(ph)

    # 2) solve state:  ∫∇y·∇v = ∫f v + ∫ z v
    l_state(v) = ∫( f*v )dΩ + ∫( zh*v )dΩ
    op_state = AffineFEOperator(a_state, l_state, Us, Vs)
    yh = solve(op_state)

    # 3) solve adjoint: ∫∇p·∇s = ∫(y-yd) s
    l_adj(s) = ∫( (yh - yd)*s )dΩ
    op_adj = AffineFEOperator(a_adj, l_adj, Ua, Va)
    ph = solve(op_adj)

    # convergence monitor (L2 of updates)
    δy = yh - yh_old
    δp = ph - ph_old
    Er = sqrt(sum(∫(δy*δy + δp*δp)*dΩ))

    @info "Iter: $iter, update norm: $Er"
  end

  # Postprocess optimal control with final adjoint
  zh = _Proj(ph)

  # Errors (same as your code)
  ey = yh - yex
  ep = ph - pex
  ez = zh - zex

  el2_yh = sqrt(sum(∫(ey⋅ey)*dΩ))
  el2_ph = sqrt(sum(∫(ep⋅ep)*dΩ))
  el2_zh = sqrt(sum(∫(ez⋅ez)*dΩz))

  dofs = num_free_dofs(Us) + num_free_dofs(Ua)

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
    el2_yh, el2_ph, el2_zh, dof, _ = variational(domain, nc, order, yex, pex, zex)
    push!(el2y, el2_yh)
    push!(el2p, el2_ph)
    push!(el2z, el2_zh)
    push!(ndofs, dof)
    push!(hs, 1/nc[1])
  end
  return el2y, el2p, el2z, hs, ndofs
end

# ---- problem data ----
yex(x) = sin(2π*x[1])*sin(2π*x[2])
pex(x) = -10*x[1]*(x[1]-1)*x[2]*(x[2]-1)

const za = -1; const zb = 1; const λ = 0.1

function _max(a,b)
  return (a + b + abs(a - b)) / 2
end

function _min(a,b)
  return (a + b - abs(a - b)) / 2
end

_Proj(p) = _max(za, _min(zb, (-1/λ) * p))
zex(x) = _Proj(pex(x))


order  = 2
domain = (0,1,0,1)
ncs = [(2,2),(4,4),(8,8),(16,16),(32,32),(64,64),(128,128),(256,256)]

el2ys, el2ps, el2zs, hs, ndofs = convg_test(domain, ncs, order, yex, pex, zex)

println("L2-error yΩ: ", el2ys)
println("Slope L2-norm state: $(slope(hs,el2ys))")

println("L2-error pΩ: ", el2ps)
println("Slope L2-norm adjoint: $(slope(hs,el2ps))")

println("L2-error zΩ: ", el2zs)
println("Slope L2-norm control: $(slope(hs,el2zs))")

el2_y, el2_p, el2_z, dofs, opt_z = variational(domain, (64,64), order, yex, pex, zex)

Ω = get_triangulation(opt_z)
writevtk(Ω, "optimal_control", cellfields=["optimal_control" => opt_z]; append=false)
