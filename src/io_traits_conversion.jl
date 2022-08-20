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
# The trait function returns a concrete type for the trait.
mixed_array_trait(::T) where {T<:AbstractArray} = NotMixed()             # Fallback
mixed_array_trait(::ArrayPartition{Union{}, Tuple{}}) = Empty()          # Covers N=0, see https://docs.julialang.org/en/v1/manual/methods/#Tuple-and-NTuple-arguments
mixed_array_trait(::ArrayPartition{<:Q, <:NTuple{N, RW(N)}}) where {N} = MatSqMut() 
#mixed_array_trait(::ArrayPartition{<:Q, <:NTuple{1, RW(1)}}) = Single()  # Covers N = 1
mixed_array_trait(::ArrayPartition{<:Q, <:NTuple{N, E}}) where {N} = N == 1 ? Single() : VecMut()
# Same as above, but called with `typeof(A)`. USeful because Julia base use this way of calling in IndexStyle()
mixed_array_trait(::Type{<:AbstractArray}) = NotMixed()             # Fallback
mixed_array_trait(::Type{<:ArrayPartition{Union{}, Tuple{}}}) = Empty()          # Covers N=0, see https://docs.julialang.org/en/v1/manual/methods/#Tuple-and-NTuple-arguments
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
"""
    convert_to_mixed(A::AbstractArray{<:Number})
    -> nested mutable ArrayPartition (Mixed)

`A` is typically Matrix{<:Number} or Vector{<:Number}.
"""
function convert_to_mixed(A::AbstractArray{<:Number})
    if is_square_matrix_mutable(A) || is_vector_mutable_stable(A)
        A
    elseif ndims(A) == 2
        @assert size(A)[1] > 1 "Cannot convert to mixed from row vectors"
        Am = ArrayPartition(map(eachrow(A)) do rw
            ArrayPartition(map(vpack, rw)...)
        end...)
        @assert Am isa MixedCandidate "Cannot convert_to_mixed from size $(size(A)) $(typeof(A))"
        Am::MixedCandidate
    elseif ndims(A) == 1
#        ArrayPartition(map(x-> [x], A)...)
         ArrayPartition(map(vpack, A)...)
    else
        throw(DimensionMismatch())
    end
end
function convert_to_mixed(A::Transpose{T, <:ArrayPartition}) where T
    apa = convert_to_array(A.parent)
    convert_to_mixed(copy(transpose(apa))) # Copy removes laziness
end

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
function convert_to_array(A)
    if A isa Array
        A
    else
        convert_to_array(mixed_array_trait(A), A)
    end
end
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
##########
# Indexing
##########
size(A::MatrixMixedCandidate) = size_of_mixed(A, mixed_array_trait(A))
size_of_mixed(A, ::MatSqMut) = size(convert_to_array(A))
size_of_mixed(A, ::T) where {T<:Mixed } = (length(A),)
size(A::AdjOrTransAbsVec{T,S}) where {T, S <: MatrixMixedCandidate} = reverse(size(A.parent))
ndims(A::MatrixMixedCandidate) = ndims_of_mixed(A, mixed_array_trait(A))
ndims_of_mixed(::MatrixMixedCandidate, ::MatSqMut) = 2
ndims_of_mixed(::AbstractArray{T,N}, ::S) where {T, N, S<:Mixed } = N
axes(A::AdjOrTransAbsVec{T,S}) where {T, S <: MatrixMixedCandidate} = reverse(axes(A.parent))

# getindex
Base.@propagate_inbounds function getindex(A::AdjOrTransAbsVec{T,S}, i::Int, j::Int) where {T, S <: MixedCandidate}
    @debug "getindex " S i j mixed_array_trait(A.parent)
    getindex_of_transposed_mixed(mixed_array_trait(A.parent), A, i, j)
end
getindex_of_transposed_mixed(::MatSqMut, A, i, j ) = A.parent[j, i]
function getindex_of_transposed_mixed(::VecMut, A, i, j )
    @assert i == 1 
    A.parent[j]
end
getindex_of_transposed_mixed(::S, A, i, j) where {S<:Mixed } = throw_boundserror(A, (i, j))


# setindex!
Base.@propagate_inbounds function setindex!(A::AdjOrTransAbsVec{T,S}, v, i::Int, j::Int) where {T, S <: MixedCandidate}
    @debug "setindex! " v i j S mixed_array_trait(A.parent)
    setindex!_of_transposed_mixed(mixed_array_trait(A.parent), A, v, i, j)
end
setindex!_of_transposed_mixed(::MatSqMut, A, v, i, j ) = setindex!(A.parent, v, j, i)
function setindex!_of_transposed_mixed(::VecMut, A, v, i, j )
    @assert i == 1
    setindex!(A.parent, v, j)
end
setindex!_of_transposed_mixed(::S, A, v, i, j) where {S<:Mixed } = throw_boundserror(A, (i, j))

# index style
# Because: IndexStyle(transpose(typeof([1 2;3 4]))) -> IndexCartesian()
# This is (likely) used by the fallback `show`` methods
function _IndexStyle(::Type{<:AdjOrTransAbsVec{T,S} where {T, S <: MixedCandidate}}) 
    @debug "_IndexStyle" 
    throw("Unused?")
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
    #printstyled(ioc, color = col, "$N-element mutable ")
    #print(ioc, "ArrayPartition(")
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



#########################################################
# Decorated (human readable) representation, trait based.
#########################################################
summary(io::IO, A::MixedCandidate) = summary(io, mixed_array_trait(A), A)
# Fallback, same as Base/show.jl:2803
summary(io::IO, ::NotMixed, A::AbstractArray) = array_summary(io, A, axes(A))
function summary(io::IO, ::MatSqMut, A)
    col = get(io, :unitsymbolcolor, :cyan)
    printstyled(io, color = col, "MatrixMixed as ")
    print(io, typeof(A))
    nothing
end
Base.summary(io::IO, ::VecMut, ::RW(N)) where {N} = print(io, "$N-element mutable ArrayPartition")
Base.summary(io::IO, ::Single, X) = print(io, "Single-element (discouraged) ArrayPartition(ArrayPartition(Vector{<:Number}))")

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


