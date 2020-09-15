
struct MockField{T,D} <: NewField
  v::T
  function MockField{D}(v::Number) where {T,D}
    new{typeof(v),D}(v)
  end
end

MockField(D::Int,v::Number) = MockField{D}(v)

mock_field(D::Int,v::Number) = MockField{D}(v)


function return_cache(f::MockField,x::AbstractArray{<:Point})
  nx = length(x)
  c = zeros(typeof(f.v),nx)
  CachedArray(c)
end

function evaluate!(c,f::MockField,x::AbstractArray{<:Point})
  nx = length(x)
  setsize!(c,(nx,))
  for i in eachindex(x)
    @inbounds xi = x[i]
    @inbounds c[i] = f.v*xi[1]
  end
  c.array
end

@inline function gradient(f::MockField{T,D}) where {T,D}
  E = eltype(T)
  P = Point{D,E}
  _p = zero(mutable(P))
  _p[1] = one(E)
  p = Point(_p)
  vg = outer(p,f.v)
  MockField{D}(vg)
end

const MockBasis{T,D,N} = AbstractArray{MockField{T,D},N}

MockBasis(v::Number,ndofs::Int) = fill(MockField(v),ndofs)

function _return_type(::MockBasis{T}) where T
  T
end

function return_cache(af::MockBasis{T},x::AbstractArray{<:Point}) where T
  np = length(x)
  s = (np, size(af)...)
  c = zeros(T,s...)
  CachedArray(c)
end

function evaluate!(v,af::MockBasis,x::AbstractArray{<:Point})
  np = length(x)
  s = (np, size(af)...)
  setsize!(v,s)
  for i in 1:np
    @inbounds xi = x[i]
    for j in CartesianIndices(af)
      @inbounds v[i,j] = af[j].v*xi[1]
    end
  end
  v.array
end

struct OtherMockField{D} <: NewField end


function return_cache(f::OtherMockField{D},x::AbstractArray{<:Point{D}}) where D
  np = length(x)
  s = (np,)
  c = zeros(eltype(x),s)
  CachedArray(c)
end

function evaluate!(v,f::OtherMockField{D},x::AbstractArray{<:Point{D}}) where D
  np = length(x)
  s = (np,)
  setsize!(v,s)
  for i in 1:np
    @inbounds v[i] = 2*x[i]
  end
  v.array
end

@inline function gradient(af::OtherMockField{D}) where D
  E = Float64
  P = Point{D,E}
  p = zero(P)
  vg = 2*one(outer(p,p))
  MockField{D}(vg)
end

const OtherMockBasis{D,N} = AbstractArray{OtherMockField{D},N}

function return_cache(af::OtherMockBasis{D},x::AbstractArray{<:Point{D}}) where D
  np = length(x)
  s = (np, size(af)...)
  c = zeros(eltype(x),s)
  CachedArray(c)
end

function evaluate!(v,af::OtherMockBasis{D},x::AbstractArray{<:Point{D}}) where D
  np = length(x)
  s = (np, size(af)...)
  setsize!(v,s)
  for i in 1:np
    @inbounds xi = x[i]
    for j in 1:f.ndofs
      @inbounds v[i,j] = 2*xi
    end
  end
  v.array
end

@inline function gradient(af::AbstractArray{OtherMockField{D}}) where D
  fg = gradient(OtherMockField{D}())
  fill(fg,size(af))
end
