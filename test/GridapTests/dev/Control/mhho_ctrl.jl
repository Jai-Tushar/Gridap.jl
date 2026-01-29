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

##############################################################

function slope(hs,errors)
  x = log10.(hs)
  y = log10.(errors)
  linreg = hcat(fill!(similar(x), 1), x) \ y
  linreg[2]
end

#######################################

# function variational_hho(domain, nc, order, yex, pex, zex, f, yd, za, zb)

  model = UnstructuredDiscreteModel(CartesianDiscreteModel(domain,nc))
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

  # State
  Vs = FESpace(Ω, reffe_V; conformity=:L2)
  Ms = FESpace(Γ, reffe_M; conformity=:L2, dirichlet_tags="boundary")
  Ls = FESpace(Ω, reffe_L; conformity=:L2)
  Ns = TrialFESpace(Ms,yex)

  # Adjoint
  Va = FESpace(Ω, reffe_V; conformity=:L2)
  Ma = FESpace(Γ, reffe_M; conformity=:L2, dirichlet_tags="boundary")
  La = FESpace(Ω, reffe_L; conformity=:L2)
  Na = TrialFESpace(Ma,pex)

  mfs = MultiField.BlockMultiFieldStyle(2,(1,1))
  Xs = MultiFieldFESpace([Vs, Ns];style=mfs)
  Ys = MultiFieldFESpace([Vs, Ms];style=mfs)
  
  Xa = MultiFieldFESpace([Va, Na];style=mfs)
  Ya = MultiFieldFESpace([Va, Ma];style=mfs)
  
  PΓs = projection_operator(Ms, Γp, dΓp)
  PΓa = projection_operator(Ma, Γp, dΓp)
  Rs  = reconstruction_operator(ptopo,Ls,Ys,Ωp,Γp,dΩp,dΓp)
  Ra  = reconstruction_operator(ptopo,La,Ya,Ωp,Γp,dΩp,dΓp)

  ph = FEFunction(Xa[1], zeros(num_free_dofs(Xa[1]))); # initial guess

  function _max(a,b)
    return (a+b+abs(a-b))/2
  end

  function _min(a,b)
    return (a+b-abs(a-b))/2
  end

  _Proj(p) = _max( za, _min( zb, (-1/λ)*p ) )

  function as(u,v)
    Ru_Ω, Ru_Γ = Rs(u)
    Rv_Ω, Rv_Γ = Rs(v)
    return ∫(∇(Ru_Ω)⋅∇(Rv_Ω) + ∇(Ru_Γ)⋅∇(Rv_Ω) + ∇(Ru_Ω)⋅∇(Rv_Γ) + ∇(Ru_Γ)⋅∇(Rv_Γ))dΩp
  end

  du, dv = get_trial_fe_basis(Xs), get_fe_basis(Ys)
  wh = FEFunction(Xs, zeros(num_free_dofs(Xs)))
  Gridap.FESpaces.get_cell_dof_values(wh)

  function aa(u,v)
    Ru_Ω, Ru_Γ = Ra(u)
    Rv_Ω, Rv_Γ = Ra(v)
    return ∫(∇(Ru_Ω)⋅∇(Rv_Ω) + ∇(Ru_Γ)⋅∇(Rv_Ω) + ∇(Ru_Ω)⋅∇(Rv_Γ) + ∇(Ru_Γ)⋅∇(Rv_Γ))dΩp
  end

  hT = maximum(sqrt(2).*sqrt.(get_array(∫(1)dΩp)))
  hTinv =  CellField(1 ./ (sqrt(2).*sqrt.(get_array(∫(1)dΩp))),Ωp)
  function ss(u,v)
    function SΓs(u)
      u_Ω, u_Γ = u
      return PΓs(u_Ω) - u_Γ
    end
    return ∫(hTinv * (SΓs(u)⋅SΓs(v)))dΓp
  end

  function sa(u,v)
    function SΓa(u)
      u_Ω, u_Γ = u
      return PΓa(u_Ω) - u_Γ
    end
    return ∫(hTinv * (SΓa(u)⋅SΓa(v)))dΓp
  end

  ls((vΩ,vΓ),p0) = ∫(f*vΩ)dΩp + ∫(_Proj(p0)*vΩ)dΩp
  la((vΩ,vΓ),y1) = ∫( (y1-yd)*vΩ )dΩp  

  function patch_weakform_s(X,Y,p0)
    u, v = get_trial_fe_basis(X), get_fe_basis(Y)
    Xp = FESpaces.PatchFESpace(X,ptopo)
    data1 = FESpaces.collect_patch_cell_matrix_and_vector(patch_assem_s,X,Y,ss(u,v),ls(v,p0),zero(X))
    data2 = FESpaces.collect_patch_cell_matrix_and_vector(patch_assem_s,Xp,Xp,as(u,v),DomainContribution(),zero(Xp))
    data = FESpaces.merge_assembly_matvec_data(data1,data2)
    return assemble_matrix_and_vector(patch_assem_s,data)
  end
  
  function patch_weakform_a(X,Y,y0)
    u, v  = get_trial_fe_basis(X), get_fe_basis(Y)
    Xp = FESpaces.PatchFESpace(X,ptopo)
    data1 = FESpaces.collect_patch_cell_matrix_and_vector(patch_assem_a,X,Y,sa(u,v),la(v,y0),zero(X))
    data2 = FESpaces.collect_patch_cell_matrix_and_vector(patch_assem_a,Xp,Xp,aa(u,v),DomainContribution(),zero(Xp))
    data  = FESpaces.merge_assembly_matvec_data(data1,data2)
    return assemble_matrix_and_vector(patch_assem_a,data)
  end

  patch_assem_s = FESpaces.PatchAssembler(ptopo,Xs,Ys)
  patch_assem_a = FESpaces.PatchAssembler(ptopo,Xa,Ya)


  tol = 1e-5
  max_iter = 50
  function solve_KKT(p0,tol,max_iter)

    iter = 0
    Er = 1.0
    yΩ = nothing
    pΩ = nothing
    yΓ = nothing
    pΓ = nothing
    while (Er > tol) && (iter < max_iter)
      iter += 1

      op_s = MultiField.StaticCondensationOperator(Xs,Vs,Ns,patch_assem_s,patch_weakform_s(Xs,Ys,p0))
      yΓ = solve(op_s.sc_op) 
      yΩ = MultiField.backward_static_condensation(op_s,yΓ)

      op_a = MultiField.StaticCondensationOperator(Xa,Va,Na,patch_assem_a,patch_weakform_a(Xa,Ya,yΩ))
      pΓ = solve(op_a.sc_op) 
      pΩ = MultiField.backward_static_condensation(op_a,pΓ)

      # Er = norm(pΩ.free_values - p0.free_values) 
      Er = sqrt( sum( ∫( (pΩ - p0)⋅(pΩ - p0) )dΩp ) )
      println("iter: $iter, Er: $Er")
      p0 = pΩ
    end

    return yΩ, pΩ, yΓ, pΓ

  end


  # optimal
  sol = solve_KKT(ph,tol,max_iter)
  yΩ, pΩ, yΓ, pΓ = sol;

  zΩ = _Proj(pΩ)

  ey = yΩ - yex
  ep = pΩ - pex
  ez = zex - zΩ

  el2yΩ = sqrt( sum( ∫(ey * ey)*dΩp ) )
  el2pΩ = sqrt( sum( ∫(ep * ep)*dΩp ) )
  el2zΩ = sqrt( sum( ∫(ez * ez)*dΩp ) )

#   return el2yΩ, el2pΩ, el2zΩ, hT
# end

function convg_test(domain, ncs, order, yex, pex, zex, f, yd, za, zb)
  el2ys = []
  el2ps = []
  el2zs = []
  hTs = []
  for n in ncs
    println(repeat("=", 50))
    println("nc = $n") 
    el2y, el2p, el2z, hT = variational_hho(domain, n, order, yex, pex, zex, f, yd, za, zb)
    push!(el2ys, el2y)
    push!(el2ps, el2p)
    push!(el2zs, el2z)
    push!(hTs, hT)
  end  
  println("L2-error yΩ: ", el2ys)
  println("L2-error pΩ: ", el2ps)
  println("L2-error zΩ: ", el2zs)
  println("mesh parameter: ", hTs)
  println("L2-slope yΩ: $(slope(hTs,el2ys))" )
  println("L2-slope pΩ: $(slope(hTs,el2ps))" )
  println("L2-slope zΩ: $(slope(hTs,el2zs))" )

  return el2ys, el2ps, el2zs, hTs
end

yex(x) = sin(2π*x[1])*sin(2π*x[2])
pex(x) = -10*sin(2*pi*x[1])*sin(2*pi*x[2])*x[1]*(x[1]-1)*x[2]*(x[2]-1)
za = -1; zb = 1; λ = 0.01
zex(x) = max( za, min(zb, -(1/λ) * pex(x)) )
f(x) = -Δ(yex)(x) - zex(x)
yd(x) = yex(x) + Δ(pex)(x)

domain = (0,1,0,1)
ncs = [(8,8),(16,16),(32,32),(64,64)]
order = 1

el2ys, el2ps, el2zs, hTs = convg_test(domain, ncs, order, yex, pex, zex, f, yd, za, zb)


function slope(hs,errors)
  x = log10.(hs)
  y = log10.(errors)
  linreg = hcat(fill!(similar(x), 1), x) \ y
  linreg[2]
end

println("Slope L2-norm state: $(slope(hTs,el2ys))")
println("Slope L2-norm adjoint: $(slope(hTs,el2ps))")
println("Slope L2-norm control: $(slope(hTs,el2zs))")