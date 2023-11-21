
"""
    struct AffineOperator{A<:AbstractMatrix,B<:AbstractVector} <: NonlinearOperator
      matrix::A
      vector::B
    end
"""
struct AffineOperator{A<:AbstractMatrix,B<:AbstractVector} <: NonlinearOperator
  matrix::A
  vector::B
end

"""
    get_matrix(operator)

Return the matrix corresponding to the assembled left hand side of the operator.
This matrix incorporates all boundary conditions and constraints.
"""
get_matrix(op::AffineOperator) = op.matrix

"""
    get_vector(operator)

Return the vector corresponding to the assembled right hand side of the operator.
This vector includes all boundary conditions and constraints.
"""
get_vector(op::AffineOperator) = op.vector

function residual!(b::AbstractVector,op::AffineOperator,x::AbstractVector)
  mul!(b,op.matrix,x)
  b .-= op.vector
  b
end

function jacobian!(A::AbstractMatrix,op::AffineOperator,x::AbstractVector)
  copy_entries!(A,op.matrix)
  A
end

function jacobian(op::AffineOperator,x::AbstractVector)
  op.matrix
end

function zero_initial_guess(op::AffineOperator)
  x = allocate_in_domain(typeof(op.vector),op.matrix)
  fill!(x,zero(eltype(x)))
  x
end

function allocate_residual(op::AffineOperator,x::AbstractVector)
  allocate_in_range(typeof(op.vector),op.matrix)
end

function allocate_jacobian(op::AffineOperator,x::AbstractVector)
  op.matrix
end

"""
    abstract type LinearSolver <: NonlinearSolver end

- [`symbolic_setup(::LinearSolver,mat::AbstractMatrix)`](@ref)
- [`test_linear_solver`](@ref)

"""
abstract type LinearSolver <: NonlinearSolver end

"""
    symbolic_setup(::LinearSolver,mat::AbstractMatrix) -> SymbolicSetup
"""
function symbolic_setup(::LinearSolver,mat::AbstractMatrix)
  @abstractmethod
end

"""
    solve(ls::LinearSolver,A::AbstractMatrix,b::AbstractVector)
"""
function solve(ls::LinearSolver,A::AbstractMatrix,b::AbstractVector)
  x = allocate_in_domain(typeof(b),A)
  solve!(x,ls,A,b)
end

"""
    solve!(x::AbstractVector,ls::LinearSolver,A::AbstractMatrix,b::AbstractVector)
"""
function solve!(x::AbstractVector,ls::LinearSolver,A::AbstractMatrix,b::AbstractVector)
  ss = symbolic_setup(ls,A)
  ns = numerical_setup(ss,A)
  solve!(x,ns,b)
  return x
end

"""
    abstract type SymbolicSetup <: GridapType end

- [`numerical_setup(::SymbolicSetup,mat::AbstractMatrix)`](@ref)
- [`numerical_setup(::SymbolicSetup,mat::AbstractMatrix,x::AbstractVector)`](@ref)
"""
abstract type SymbolicSetup <: GridapType end

"""
    numerical_setup(::SymbolicSetup,mat::AbstractMatrix) -> NumericalSetup
"""
function numerical_setup(::SymbolicSetup,mat::AbstractMatrix)
  @abstractmethod
end

"""
    numerical_setup(::SymbolicSetup,mat::AbstractMatrix,x::AbstractVector) -> NumericalSetup
"""
function numerical_setup(ss::SymbolicSetup,mat::AbstractMatrix,x::AbstractVector)
  numerical_setup(ss,mat)
end

"""
    abstract type NumericalSetup <: GridapType end

- [`numerical_setup!(ns::NumericalSetup,mat::AbstractMatrix)`](@ref)
- [`numerical_setup!(ns::NumericalSetup,mat::AbstractMatrix,x::AbstractVector)`](@ref)
- [`solve!(x::AbstractVector,ns::NumericalSetup,b::AbstractVector)`](@ref)
"""
abstract type NumericalSetup <: GridapType end

"""
    numerical_setup!(ns::NumericalSetup,mat::AbstractMatrix)
"""
function numerical_setup!(ns::NumericalSetup,mat::AbstractMatrix)
  @abstractmethod
end

"""
    numerical_setup!(ns::NumericalSetup,mat::AbstractMatrix,x::AbstractVector)
"""
function numerical_setup!(ns::NumericalSetup,mat::AbstractMatrix,x::AbstractVector)
  numerical_setup!(ns,mat)
end

"""
    solve!(x::AbstractVector,ns::NumericalSetup,b::AbstractVector)
"""
function solve!(::AbstractVector,::NumericalSetup,::AbstractVector)
  @abstractmethod
end

"""
    test_linear_solver(
      ls::LinearSolver,
      A::AbstractMatrix,
      b::AbstractVector,
      x::AbstractVector)
"""
function test_linear_solver(
  ls::LinearSolver,
  A::AbstractMatrix,
  b::AbstractVector,
  x::AbstractVector)

  y = solve(ls,A,b)
  @test x ≈ y

  fill!(y, zero(eltype(y)))
  ss = symbolic_setup(ls,A)
  ns = numerical_setup(ss,A)
  numerical_setup!(ns,A)
  solve!(y,ns,b)
  @test x ≈ y

  fill!(y, zero(eltype(y)))
  solve!(y,ls,A,b)
  @test x ≈ y

  op = AffineOperator(A,b)
  x0 = copy(x)
  test_nonlinear_solver(ls,op,x0,x)

end

struct LinearSolverCache <: GridapType
  A::AbstractMatrix
  b::AbstractVector
  ns::NumericalSetup
end

function solve!(x::AbstractVector,
                ls::LinearSolver,
                op::NonlinearOperator,
                cache::Nothing)
  fill!(x,zero(eltype(x)))
  b = residual(op, x)
  A = jacobian(op, x)
  ss = symbolic_setup(ls, A)
  ns = numerical_setup(ss,A)
  rmul!(b,-1)
  solve!(x,ns,b)
  LinearSolverCache(A,b,ns)
end

function solve!(x::AbstractVector,
                ls::LinearSolver,
                op::NonlinearOperator,
                cache)
  fill!(x,zero(eltype(x)))
  b = cache.b
  ns = cache.ns
  residual!(b, op, x)
  numerical_setup!(ns,cache.A)
  rmul!(b,-1)
  solve!(x,ns,b)
  cache
end

function solve!(x::AbstractVector,ls::LinearSolver,op::AffineOperator,cache::Nothing)
  A  = get_matrix(op)
  b  = get_vector(op)
  ss = symbolic_setup(ls,A)
  ns = numerical_setup(ss,A)
  solve!(x,ns,b)
  ns
end

function solve!(x::AbstractVector,ls::LinearSolver,op::AffineOperator,cache)
  newmatrix = true
  solve!(x,ls,op,cache,newmatrix)
  cache
end

"""
    solve!(
      x::AbstractVector,
      ls::LinearSolver,
      op::AffineOperator,
      cache,
      newmatrix::Bool)
"""
function solve!(x::AbstractVector,ls::LinearSolver,op::AffineOperator,cache,newmatrix::Bool)
  A = op.matrix
  b = op.vector
  ns = cache
  if newmatrix
    numerical_setup!(ns,A)
  end
  solve!(x,ns,b)
  cache
end

function solve!(x::AbstractVector,ls::LinearSolver,op::AffineOperator,cache::Nothing,newmatrix::Bool)
  solve!(x,ls,op,cache)
end

# Concrete implementations

"""
    struct LUSolver <: LinearSolver end

Wrapper of the LU solver available in julia
"""
struct LUSolver <: LinearSolver end

struct LUSymbolicSetup <: SymbolicSetup end

mutable struct LUNumericalSetup{F} <: NumericalSetup
  factors::F
end

symbolic_setup(::LUSolver,mat::AbstractMatrix) = LUSymbolicSetup()

numerical_setup(::LUSymbolicSetup,mat::AbstractMatrix) = LUNumericalSetup(lu(mat))

function numerical_setup!(ns::LUNumericalSetup, mat::AbstractMatrix)
  fac = lu(mat)
  ns.factors = fac
  ns
end

function numerical_setup!(ns::LUNumericalSetup, mat::SparseMatrixCSC)
  lu!(ns.factors,mat)
  ns
end

function solve!(
  x::AbstractVector,ns::LUNumericalSetup,b::AbstractVector)
  ldiv!(x,ns.factors,b)
  x
end

"""
    struct BackslashSolver <: LinearSolver end

Wrapper of the backslash solver available in julia
This is typically faster than LU for a single solve
"""
struct BackslashSolver <: LinearSolver end

struct BackslashSymbolicSetup <: SymbolicSetup end

mutable struct BackslashNumericalSetup{T<:AbstractMatrix} <: NumericalSetup
  A::T
end

symbolic_setup(::BackslashSolver,mat::AbstractMatrix) = BackslashSymbolicSetup()

numerical_setup(::BackslashSymbolicSetup,mat::AbstractMatrix) = BackslashNumericalSetup(mat)

function numerical_setup!(ns::BackslashNumericalSetup, mat::AbstractMatrix)
  ns.A = mat
end

function solve!(
  x::AbstractVector,ns::BackslashNumericalSetup,b::AbstractVector)
  copy_entries!(x, ns.A\b)
end
