###############################################
# Inferrable mixed matrices during broadcasting
###############################################
# Types and traits are defined in 'io_traits_conversion'
# These definitions tells the compiler more type
# details in advance of mapping or broadcasting.

# Overload MixedMatrices - numer of axes always 2
function Broadcast.BroadcastStyle(::Type{ArrayPartition{T,S}}) where {T, S<:NTuple{N, RW(N)} where N}
    ArrayPartitionStyle(Base.Broadcast.DefaultArrayStyle{2}())
end

@inline combine_styles(A::MixedCandidate, B::AbstractMatrix) = _combine_styles(mixed_array_trait(A), A, B)
@inline combine_styles(A::AbstractMatrix, B::MixedCandidate) = _combine_styles(mixed_array_trait(B), B, A)
@inline _combine_styles(::MatSqMut, A, _) = combine_styles(A)
@inline _combine_styles(::Mixed, A, B) = combine_styles(A)                       # Fallback

@inline function Base.copy(bc::Broadcast.Broadcasted{ArrayPartitionStyle{Base.Broadcast.DefaultArrayStyle{2}}})
    @inline function outer(i)
        copy(broadcasted(bc.f, unpack_args(i, bc.args)...))
    end
    ArrayPartition(ntuple(outer, Val(npartitions(bc))))
end

# For broadcasting mixed matrices  with normal matrices
unpack(x::AbstractMatrix, i) = ArrayPartition(map(vpack, x[i, :])...)


## Iterable Collection Constructs
# 2-argument map not inferrable. To improve on this, see the implementations for SparseArrays or StaticArrays.
Base.map(f, A::MixedCandidate, B::MixedCandidate) = _map(mixed_array_trait(A),f, A, B)
Base.map(f, A::MixedCandidate, B::AbstractMatrix) = _map(mixed_array_trait(A),f, A, B)
Base.map(f, A::AbstractMatrix, B::MixedCandidate) = _map(mixed_array_trait(B),f, A, B)
function _map(::MatSqMut, f, A, B)
    ArrayPartition(map(zip(eachrow(A), eachrow(B))) do (rwA, rwB)
        maprow(f, rwA, rwB)
    end...)
end
_map(::VecMut, f, iters...) = collect(Base.Generator(f, iters...))

function maprow(f, rwA, rwB)
    ArrayPartition(map(zip(rwA, rwB)) do (a, b)
        vpack(f(a, b))
    end...)
end
zip(A::MixedCandidate, B::AbstractMatrix) = _zip(mixed_array_trait(A), A, B)
zip(A::AbstractMatrix, B::MixedCandidate) = _zip(mixed_array_trait(B), A, B)
_zip(::MatSqMut, A::MixedCandidate, B) = Base.Iterators.Zip((A, transpose(B)))
_zip(::MatSqMut, A, B::MixedCandidate) = Base.Iterators.Zip((transpose(A), B))
_zip(::Mixed, A, B) = Base.Iterators.Zip((A, B))                   # Fallback

######################################
# Inferrable getindex
# Broadcasting is already inferrable. 
# These methods improves other 
# inferrence by inlining. 
# Tested in test_014.jl
######################################
# Extends RecursiveArrayTools\src\array_partition.jl:200
Base.@propagate_inbounds @inline Base.getindex(A::MixedCandidate, i::Int) = _getindex(mixed_array_trait(A), A, i)
@inline _getindex(::VecMut, A::ArrayPartition, i::Int) = first(getindex(A.x, i)) 
@inline function _getindex(::VecMut, A::ArrayPartition, i::Int, j)
    @boundscheck 0 < i <= length(A.x) || throw(BoundsError(A.x, i))
    @boundscheck j == 1 || throw(BoundsError(A.x, j))
    first(getindex(A.x, i))
end
@inline function _getindex(::Single, A::ArrayPartition, i::Int)
    @boundscheck i == 1 || throw(BoundsError(A.x, i))
    first(getindex(A.x, i))
end
@inline function _getindex(::Single, A::ArrayPartition, i::Int, j)
    @boundscheck i == 1 || throw(BoundsError(A.x, i))
    @boundscheck j == 1 || throw(BoundsError(A.x, j))
    first(getindex(A.x, i))
end
# Extends RecursiveArrayTools\src\array_partition.jl:215
Base.@propagate_inbounds @inline Base.getindex(A::MixedCandidate, i::Int, j::Int) = _getindex(mixed_array_trait(A), A, i, j)
@inline function _getindex(::MatSqMut, A::ArrayPartition, i::Int, j::Int)
    @boundscheck 0 < i <= length(A.x) || throw(BoundsError(A.x, i))
    @inbounds b = A.x[i]
    @boundscheck checkbounds(b, j)
    @inbounds return first(b.x[j])
end

#######
# TODO
#######

# Todo consider implementing for mixed matrices:
#Base.mapreduce(f,op,A::ArrayPartition) = mapreduce(f,op,(mapreduce(f,op,x) for x in A.x))
#Base.filter(f,A::ArrayPartition) = ArrayPartition(map(x->filter(f,x), A.x))
#Base.any(f,A::ArrayPartition) = any(f,(any(f,x) for x in A.x))
#Base.any(f::Function,A::ArrayPartition) = any(f,(any(f,x) for x in A.x))
#Base.any(A::ArrayPartition) = any(identity, A)
#Base.all(f,A::ArrayPartition) = all(f,(all(f,x) for x in A.x))
#Base.all(f::Function,A::ArrayPartition) = all(f,(all(f,x) for x in A.x))
#Base.all(A::ArrayPartition) = all(identity, A)

