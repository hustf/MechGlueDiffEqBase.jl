###############
# Types, traits
###############

# We want to specialize on ArrayPartitions that
# - can represent square matrices
# - are mutable
# - have width and height > 1
# - have elements that are Real, Complex, Quantity{<:Real}, Quantity{<:Complex}
# - are inferrable
# ...because Matrix{Any} is mostly not inferrable. For inferrability, we implement
# such matrix-like objects as nested ArrayPartition.
# For mutability, the innermost type is a one-element vector.
# A value (consider Float64):
const Q = Number
"A mutable element Vector{<:Q}"
const E = Vector{<:Q}
"""
    RW(N) = ArrayPartition{<:Q, <:NTuple{N, E}}
Shorthand 'meta function'
Example

    foo(v::RW(N)) where N = 1
"""
const RW(N) = ArrayPartition{<:Q, <:NTuple{N, E}}

"""
This imperfect definition of objects we want to dispatch on includes immutable
versions and empty tuples. It dispatches a bit too widely (includes empty tuples),
so we use traits-based dispatch with this type:
"""
const MatrixMixedCandidate = ArrayPartition{<:Q, <:NTuple{N, RW(N)}} where {N}
const MixedContent(N) = ArrayPartition{<:Q, <:NTuple{N, Union{RW(N), E}}}
const MixedCandidate = MixedContent(N) where {N}
# Traits are concrete types <: Mixed
abstract type Mixed end
struct MatSqMut <: Mixed end
struct VecMut <: Mixed end
struct NotMixed <: Mixed end
struct Single <: Mixed end
struct Empty <: Mixed end
const UnionVecSqMut = Union{VecMut, MatSqMut}
# The trait function returns a concrete type for the trait.
mixed_array_trait(::T) where {T<:AbstractArray} = NotMixed()             # Fallback
mixed_array_trait(::ArrayPartition{Union{}, Tuple{}}) = Empty()          # Covers N = 0, see https://docs.julialang.org/en/v1/manual/methods/#Tuple-and-NTuple-arguments
mixed_array_trait(::ArrayPartition{<:Q, <:NTuple{N, RW(N)}}) where {N} = MatSqMut()
#mixed_array_trait(::ArrayPartition{<:Q, <:NTuple{1, RW(1)}}) = Single()  # Covers N = 1
mixed_array_trait(::ArrayPartition{<:Q, <:NTuple{N, E}}) where {N} = N == 1 ? Single() : VecMut()
# Same as above, but called with `typeof(A)`. USeful because Julia base use this way of calling in IndexStyle()
mixed_array_trait(::Type{<:AbstractArray}) = NotMixed()             # Fallback
mixed_array_trait(::Type{<:ArrayPartition{Union{}, Tuple{}}}) = Empty()          # Covers N = 0, see https://docs.julialang.org/en/v1/manual/methods/#Tuple-and-NTuple-arguments
mixed_array_trait(::Type{<:ArrayPartition{<:Q, <:NTuple{N, RW(N)} where {N}}}) = MatSqMut()
#mixed_array_trait(::Type{<:ArrayPartition{<:Q, <:NTuple{1, RW(1)}}}) = Single()  # Covers N = 1
mixed_array_trait(::Type{<:ArrayPartition{<:Q, <:NTuple{N, E}}}) where {N} = N == 1 ? Single() : VecMut()

is_square_matrix_mutable(M) = mixed_array_trait(M) isa MatSqMut
is_vector_mutable_stable(v) = mixed_array_trait(v) isa VecMut


############
# Conversion
############
vpack(x) = [x]
vpack(x::DimensionlessQuantity) = [uconvert(NoUnits, x)]
function vpack(x::AbstractVector)
    @assert length(x) == 1
    vpack(first(x))
end

"""
    convert_to_mixed(x...)
    -> nested mutable ArrayPartition (Mixed).
    Vector-like or Matrix-like depending on input.
"""
function convert_to_mixed(A::AbstractArray)
    if ndims(A) == 2
        @assert size(A)[1] > 1 "Cannot convert to mixed from row vectors"
        Am = convert_to_mixed(eachrow(A))
        @assert Am isa MixedCandidate "Cannot convert_to_mixed from size $(size(A)) $(typeof(A))"
        @assert mixed_array_trait(Am) == MatSqMut()
        Am
    elseif ndims(A) == 1
        # Not tested, vector types are expected to dispatch elsewhere.
        convert_to_mixed(A...)
    else
        throw(DimensionMismatch())
    end
end
convert_to_mixed(x::AbstractVector{T}) where T = convert_to_mixed(x...)
convert_to_mixed(x::NTuple{N, U}) where {N, U} = ArrayPartition_from_single_element_vectors(vpack.(x))
# Previously dispatched on: x::Vararg{<:Number}, 
# which gives the deprecation warning:  Wrapping `Vararg` directly in UnionAll is deprecated (wrap the tuple instead).
function convert_to_mixed(x::Number...)
    p = vpack.(x)
    if length(x) > 1
        ArrayPartition_from_single_element_vectors(convert.(E, p))
    else
        if length(first(p)) == 1
            ArrayPartition_from_single_element_vectors(p)
        else
            convert_to_mixed(p...)
        end
    end
end
function convert_to_mixed(x::MixedCandidate)
    @assert is_square_matrix_mutable(x) || is_vector_mutable_stable(x)
    x
end
function convert_to_mixed(A::Transpose{T, <:ArrayPartition}) where T
    apa = convert_to_array(A.parent)
    convert_to_mixed(copy(transpose(apa))) # Copy removes laziness
end
# For inferrable matrix-like
function convert_to_mixed(g::Base.Generator)
    thismixed(rw) = convert_to_mixed(rw...)
    ArrayPartition(thismixed.(g)...)
end
ArrayPartition_from_single_element_vectors(x::NTuple{N, E}) where N = ArrayPartition(x)

"""
    convert_to_array(A::ArrayPartition)
    --> Matrix{Any} or Vector{Any}.

We avoid overloading base function ´convert´on types defined by ArrayTools.

    # Examples
    ```julia-repl
    julia> M2u = ArrayPartition(ArrayPartition([1]kg, [2]s), ArrayPartition([3]s, [4]kg));

    julia> convert_to_array(M2u)

    julia> Vu = ArrayPartition([1.0]s⁻¹, [2.0]s⁻²)

    julia> convert_to_array(Vu)

    ```
Note the row-first order of M2:
# Example
```julia-repl
julia> reshape(convert(Array{Any}, M2), (2, 2))'
```

"""
convert_to_array(A::Array) = A
convert_to_array(A) = convert_to_array(mixed_array_trait(A), A)

function convert_to_array(::MatSqMut, A::ArrayPartition{T, S}) where {T, S}
    m = length(A.x)
    n = length(A.x[1])
    X = Array{T, 2}(undef, m, n)
    for i = 1:m
        # Square. Can easily be dropped to extend functionality.
        @assert n == length(A.x[i])
        for j = 1:n
            X[i, j] = A.x[i][j]
        end
    end
    X
end
function convert_to_array(::VecMut, A::ArrayPartition{T, S}) where {T, S}
    Vector{T}(A)
end
function convert_to_array(::NotMixed, A::Transpose{T, <:ArrayPartition}) where {T}
    apa = convert_to_array(A.parent)
    copy(transpose(apa)) # Copy removes laziness
end
function convert_to_array(::Mixed, A)
    throw(InexactError(:convert_to_array, Array, A))
    A
end
################################################
# Indexing (more in 'broadcast_mixed_matrix.jl')
################################################
size(A::MatrixMixedCandidate) = size_of_mixed(A, mixed_array_trait(A))
size_of_mixed(A, ::MatSqMut) = size(convert_to_array(A))
size_of_mixed(A, ::T) where {T<:Mixed } = (length(A),)
size(A::AdjOrTransAbsVec{T,S}) where {T, S <: MatrixMixedCandidate} = reverse(size(A.parent))
ndims(A::MatrixMixedCandidate) = ndims_of_mixed(A, mixed_array_trait(A))
ndims_of_mixed(::MatrixMixedCandidate, ::MatSqMut) = 2
ndims_of_mixed(::AbstractArray{T,N}, ::S) where {T, N, S<:Mixed } = N
axes(A::AdjOrTransAbsVec{T,S}) where {T, S <: MatrixMixedCandidate} = reverse(axes(A.parent))

# getindex
Base.@propagate_inbounds @inline function getindex(A::AdjOrTransAbsVec{T,S}, i::Int, j::Int) where {T, S <: MixedCandidate}
    getindex_of_transposed_mixed(mixed_array_trait(A.parent), A, i, j)
end
@inline getindex_of_transposed_mixed(::MatSqMut, A, i, j ) = A.parent[j, i]
@inline function getindex_of_transposed_mixed(::VecMut, A, i, j )
    i !== 1 && throw_boundserror(A, (i, j))
    A.parent[j]
end
@inline getindex_of_transposed_mixed(::S, A, i, j) where {S<:Mixed } = throw_boundserror(A, (i, j))


# setindex!
Base.@propagate_inbounds @inline function setindex!(A::AdjOrTransAbsVec{T,S}, v, i::Int, j::Int) where {T, S <: MixedCandidate}
    setindex!_of_transposed_mixed(mixed_array_trait(A.parent), A, v, i, j)
end
@inline setindex!_of_transposed_mixed(::MatSqMut, A, v, i, j ) = setindex!(A.parent, v, j, i)
@inline function setindex!_of_transposed_mixed(::VecMut, A, v, i, j )
    @assert i == 1
    setindex!(A.parent, v, j)
end
@inline setindex!_of_transposed_mixed(::S, A, v, i, j) where {S<:Mixed } = throw_boundserror(A, (i, j))

# index style
# Because: IndexStyle(transpose(typeof([1 2;3 4]))) -> IndexCartesian()
# This is (likely) used by the fallback `show`` methods
function _IndexStyle(::Type{<:AdjOrTransAbsVec{T,S} where {T, S <: MixedCandidate}})
    IndexStyle_of_transposed_mixed(mixed_array_trait(A.parent))
end
IndexStyle_of_transposed_mixed(::MatSqMut) = IndexCartesian()
IndexStyle_of_transposed_mixed(::S) where {S<:Mixed} = IndexLinear()


##########################
# IO nested ArrayPartition
##########################


######################################################
# Un-decorated (parseable) representation, trait-based
######################################################
print(io::IO, A::MixedCandidate) =  print_as_mixed(io, mixed_array_trait(A), A)
function print_as_mixed(io::IO, ::MatSqMut, A)
    X = convert_to_array(A)
    col = get(io, :unitsymbolcolor, :cyan)
    buf = IOBuffer()
    ioc = IOContext(buf, IOContext(io).dict)
    printstyled(ioc, color = col, "convert_to_mixed(")
    prefix = String(take!(buf))
    printstyled(ioc, color = col, ")")
    postfix = String(take!(buf))
    if isempty(X)
        Base._show_empty(io, X)
    else
        Base._show_nonempty(io, X, prefix)
        print(io, postfix)
    end
end
function print_as_mixed(io::IO, ::VecMut, v::RW(N)) where N
    # A normal vector would have no type info here. We provide a short summary with coloured
    # highlighting to indicate that this is not a quite normal vector.
    buf = IOBuffer()
    ioc = IOContext(buf, IOContext(io).dict)
    col = get(ioc, :unitsymbolcolor, :cyan)
    printstyled(ioc, color = col, "convert_to_mixed(")
    prefix = String(take!(buf))
    printstyled(ioc, color = col, ")")
    postfix = String(take!(buf))
    Base.show_vector(io::IO, v, prefix, postfix)
end

function print_as_mixed(io::IO, ::Single, v)
    # A normal vector would have no type info here. We provide a short summary with coloured
    # highlighting to indicate that this is not a quite normal vector.
    buf = IOBuffer()
    ioc = IOContext(buf, IOContext(io).dict)
    col = get(ioc, :unitsymbolcolor, :cyan)
    printstyled(ioc, color = col, "Single-element mutable matrix (discouraged) ")
    print(ioc, "ArrayPartition(ArrayPartition([")
    prefix = String(take!(buf))
    Base.show_vector(io::IO, v, prefix, "]))")
end

print_as_mixed(io::IO, ::Union{NotMixed, Empty}, v) = invoke(print, Tuple{typeof(io), Any}, io, v)

# Overloads RecursiveArrayTools.jl:26, which invokes showing this as 'Any'..., with too much header info:
Base.show(io::IO, A::MixedCandidate) = Base.show(io, mixed_array_trait(A), A)
function Base.show(io::IO, ::MatSqMut, A::MixedCandidate)
    # A normal matrix would have no type info here. We'll just provide a short summary with coloured
    # highlighting to indicate that this is not a quite normal matrix.
    col = get(io, :unitsymbolcolor, :cyan)
    printstyled(io, color = col, "convert_to_mixed(")
    X = convert_to_array(A)
    show(io, X)
    printstyled(io, color = col, ")")
end
function Base.show(io::IO, ::VecMut, v::RW(N)) where N
    # A normal vector would have no type info here. We provide a short summary with coloured
    # highlighting to indicate that this is not a quite normal vector.
    buf = IOBuffer()
    ioc = IOContext(buf, IOContext(io).dict)
    col = get(ioc, :unitsymbolcolor, :cyan)
    printstyled(ioc, color = col, "$N-element mutable ")
    print(ioc, "ArrayPartition(")
    prefix = String(take!(buf))
    Base.show_vector(io::IO, v, prefix, ')')
end
function Base.show(io::IO, ::Single, v)
    buf = IOBuffer()
    ioc = IOContext(buf, IOContext(io).dict)
    col = get(ioc, :unitsymbolcolor, :cyan)
    printstyled(ioc, color = col, "Single-element mutable matrix (discouraged) ")
    print(ioc, "ArrayPartition(ArrayPartition([")
    prefix = String(take!(buf))
    Base.show_vector(io::IO, v, prefix, "]))")
end
# Same as RecursiveArrayTools.jl:26
Base.show(io::IO, ::Union{NotMixed, Empty}, A::MixedCandidate) = invoke(show, Tuple{typeof(io), Any}, io, A)



##############################################
# Decorated (human readable but not parseable) 
# representation, trait based
##############################################

# We extend 'Base.summary' because extending 'Base.array_summary' would be ambiguous.
summary(io::IO, A::MixedCandidate) = _summary(io, mixed_array_trait(A), A)
# Fallback, MixedCandidate spanned too wide.
_summary(io::IO, ::NotMixed, A::AbstractArray) = invoke(array_summary, Tuple{typeof(io), Any}, io, v)
# Mixed vectors, matrices, singles:
_summary(io::IO, t::Mixed, a) = _array_summary(io, t, a, axes(a))

function trait_summary(io, t::Mixed)
    col = get(io, :unitsymbolcolor, :cyan)
    printstyled(io, color = col, "⊲ ", typeof(t), " ")
end
function _array_summary(io::IO, t::Mixed, a, inds::Tuple{Vararg{Base.OneTo}})
    print(io, Base.dims2string(length.(inds)), " ")
    trait_summary(io, t)
    # Should we display the full type info?
    sz = Base.displaysize(io)::Tuple{Int,Int}
    remainwidth = sz[2]  - 1 - 5 - 10 # margin, size, trait string
    typelength = Base.alignment(io, typeof(a))[2]
    if typelength < remainwidth
        print(io, typeof(a))
    else
        printstyled(io, "(alias:) "; color = :light_black)
        print(io, "ArrayPartition{<:Number, <:NTuple{N, Union{RW(N), E}}}")
    end
    nothing
end
# We extend 'Base.show' because RecursiveArrayTools already does that, and in a rough way..
# Otherwise, we wouldn't have to extend.
Base.show(io::IO, m::MIME"text/plain", A::MixedCandidate) = Base.show(io, m, mixed_array_trait(A), A)
function Base.show(io::IO, ::MIME"text/plain", ::MatSqMut, A::MixedCandidate)
    # 0) show summary before setting :compact
    summary(io, A)
    isempty(A) && return
    print(io, ":")
    Base.show_circular(io, A) && return
    X = convert_to_array(A)
    # 1) compute new IOContext
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end

    if get(io, :limit, false) && displaysize(io)[1]-4 <= 0
        return print(io, " …")
    else
        println(io)
    end

    # 2) Don't update typeinfo (number type not shown in header)
    # io = IOContext(io, :typeinfo => eltype(X))

    # 3) show actual content
    recur_io = IOContext(io, :SHOWN_SET => X)
    Base.print_array(recur_io, X)
end
function Base.show(io::IO, m::MIME"text/plain", ::VecMut, X::RW(N)) where N
    # 0) show summary before setting :compact
    summary(io, X)
    isempty(X) && return
    print(io, ":")
    Base.show_circular(io, X) && return

    # 1) compute new IOContext
    if get(io, :limit, false) && displaysize(io)[1]-4 <= 0
        return print(io, " …")
    else
        println(io)
    end

    # 2) Don't update typeinfo (number type not shown in header)
    # io = IOContext(io, :typeinfo => eltype(X))

    # 3) show actual content
    recur_io = IOContext(io, :SHOWN_SET => X)
    Base.print_array(recur_io, X)
end
function Base.show(io::IO, m::MIME"text/plain", ::Single, X)
    # 0) show summary before setting :compact
    summary(io, X)
    isempty(X) && return
    print(io, ":")
    Base.show_circular(io, X) && return

    # 1) compute new IOContext
    if get(io, :limit, false) && displaysize(io)[1]-4 <= 0
        return print(io, " …")
    else
        println(io)
    end

    # 2) Don't update typeinfo (number type not shown in header)
    # io = IOContext(io, :typeinfo => eltype(X))

    # 2) show actual content
    recur_io = IOContext(io, :SHOWN_SET => X)
    Base.print_array(recur_io, X)
end

# Fallback, same as \RecursiveArrayTools\tU7uv\src\array_partition.jl:248
Base.show(io::IO, m::MIME"text/plain", ::Union{NotMixed, Empty}, A::MixedCandidate) = show(io, m, A.x)


