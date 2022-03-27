# This Map has non-trivial domain, thus we need the define testargs
"""
    Reindex(values) -> Map
"""
struct Reindex{A} <: Map
  values::A
end

function testargs(k::Reindex,i)
  @check length(k.values) !=0 "This map has empty domain"
  (one(i),)
end
function testargs(k::Reindex,i::Integer...)
  @check length(k.values) !=0 "This map has empty domain"
  map(one,i)
end
function return_value(k::Reindex,i...)
  length(k.values)!=0 ? evaluate(k,testargs(k,i...)...) : testitem(k.values)
end
return_cache(k::Reindex,i...) = array_cache(k.values)
evaluate!(cache,k::Reindex,i...) = getindex!(cache,k.values,i...)
evaluate(k::Reindex,i...) = k.values[i...]

#"""
#    reindex(i_to_v::AbstractArray, j_to_i::AbstractArray)
#"""
#function reindex(i_to_v::AbstractArray, j_to_i::AbstractArray)
#  lazy_map(Reindex(i_to_v),j_to_i)
#end

function lazy_map(k::Reindex{<:Fill},::Type{T}, j_to_i::AbstractArray) where T
  v = k.values.value
  Fill(v,size(j_to_i)...)
end

function lazy_map(k::Reindex{<:CompressedArray},::Type{T}, j_to_i::AbstractArray) where T
  i_to_v = k.values
  values = i_to_v.values
  ptrs = lazy_map(Reindex(i_to_v.ptrs),j_to_i)
  CompressedArray(values,ptrs)
end

function lazy_map(k::Reindex{<:LazyArray},::Type{T},j_to_i::AbstractArray) where T
  i_to_maps = k.values.maps
  i_to_args = k.values.args
  j_to_maps = lazy_map(Reindex(i_to_maps),eltype(i_to_maps),j_to_i)
  j_to_args = map(i_to_fk->lazy_map(Reindex(i_to_fk),eltype(i_to_fk),j_to_i), i_to_args)
  LazyArray(T,j_to_maps,j_to_args...)
end

# This optimization is important for surface-coupled problems
function lazy_map(k::Reindex{<:LazyArray{<:Fill{<:PosNegReindex}}},::Type{T},j_to_i::AbstractArray) where T
  i_to_iposneg = k.values.args[1]
  ipos_to_value = k.values.maps.value.values_pos
  ineg_to_value = k.values.maps.value.values_neg
  if aligned_with_pos(i_to_iposneg,j_to_i,length(ipos_to_value))
    ipos_to_value
  elseif aligned_with_neg(i_to_iposneg,j_to_i,length(ineg_to_value))
    ineg_to_value
  elseif all_in_pos(i_to_iposneg,j_to_i)
    j_to_ipos = lazy_map(Reindex(get_array(i_to_iposneg)),j_to_i)
    j_to_value = lazy_map(Reindex(ipos_to_value),j_to_ipos)
  elseif all_in_neg(i_to_iposneg,j_to_i)
    j_to_ineg = lazy_map(Reindex(get_array(i_to_iposneg)),j_to_i)
    j_to_value = lazy_map(Reindex(ineg_to_value),lazy_map(ineg->-ineg,j_to_ineg))
  else
    j_to_iposneg = lazy_map(Reindex(get_array(i_to_iposneg)),j_to_i)
    j_to_value = lazy_map(PosNegReindex(ipos_to_value,ineg_to_value),j_to_iposneg)
  end
end

function _lazy_map_evaluate_posneg(
  i_to_jposneg::LazyArray{<:Fill{<:Reindex}},::Type{T},i_a,i_x::Fill) where T
  i_y = CompressedArray([i_x.value],Fill(Int32(1),length(i_x)))
  _lazy_map_evaluate_posneg(i_to_jposneg,T,i_a,i_y)
end

function _lazy_map_evaluate_posneg(i_to_jposneg::LazyArray{<:Fill{<:Reindex}},::Type{T},i_a,i_x::CompressedArray) where T
  jpos_a = i_a.maps.value.values_pos
  jneg_a = i_a.maps.value.values_neg
  j_jposneg = i_to_jposneg.maps.value.values
  jpos_j, jneg_j = pos_and_neg_indices(j_jposneg)
  k_x = deepcopy(i_x.values)
  i_k = i_x.ptrs
  push!(k_x,testitem(i_x))
  i_j = i_to_jposneg.args[1]
  j_k = fill(Int32(length(k_x)),length(j_jposneg))
  j_k[i_j] .= i_k
  jpos_k = j_k[jpos_j]
  jneg_k = j_k[jneg_j]
  jpos_x = CompressedArray(k_x,jpos_k)
  jneg_x = CompressedArray(k_x,jneg_k)
  jpos_b = lazy_map(evaluate,jpos_a,jpos_x)
  jneg_b = lazy_map(evaluate,jneg_a,jneg_x)
  j_b = lazy_map(PosNegReindex(jpos_b,jneg_b),j_jposneg)
  i_b = lazy_map(Reindex(j_b),i_j)
  i_b
end

function lazy_map(k::Reindex{<:LazyArray{<:Fill{<:PosNegReindex}}},::Type{T},indices::IdentityVector) where T
  @check length(k.values) == length(indices)
  k.values
end

function lazy_map(k::Reindex{<:LazyArray},::Type{T},indices::IdentityVector) where T
  @check length(k.values) == length(indices)
  k.values
end

function lazy_map(k::Reindex{<:AbstractArray},::Type{T},indices::IdentityVector) where T
  @check length(k.values) == length(indices)
  k.values
end

function lazy_map(k::Reindex{<:Fill},::Type{T},indices::IdentityVector) where T
  @check length(k.values) == length(indices)
  k.values
end

function lazy_map(k::Reindex{<:CompressedArray},::Type{T},indices::IdentityVector) where T
  @check length(k.values) == length(indices)
  k.values
end

@propagate_inbounds function Base.setindex!(a::LazyArray{<:Fill{<:Reindex}},v,j::Integer)
  k = a.maps.value
  i_to_v = k.values
  j_to_i, = a.args
  i = j_to_i[j]
  i_to_v[i]=v
end
