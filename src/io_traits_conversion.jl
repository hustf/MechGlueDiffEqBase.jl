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
RW(N) = ArrayPartition{<:Q, <:NTuple{N, E}}

"""
This imperfect definition of objects we want to dispatch on includes immutable
versions and empty tuples. It dispatches a bit too widely (includes empty tuples),
so we use traits-based dispatch with this type:
"""
const MatrixMixedCandidate = ArrayPartition{<:Q, <:NTuple{N, RW(N)}} where {N}
abstract type MatrixMixed end
struct MatSqMut <: MatrixMixed end
struct NotMatSqMut <: MatrixMixed end
# The trait function returns a concrete type for the trait.
mixed_array_trait(::T) where {T<:AbstractArray} = NotMatSqMut()             # Fallback
mixed_array_trait(::ArrayPartition{Union{}, Tuple{}}) = NotMatSqMut()       # Covers N=0, see https://docs.julialang.org/en/v1/manual/methods/#Tuple-and-NTuple-arguments
mixed_array_trait(::ArrayPartition{<:Q, <:NTuple{N, RW(N)}}) where {N} = MatSqMut() 
mixed_array_trait(::ArrayPartition{<:Q, <:NTuple{1, RW(1)}}) = NotMatSqMut() # Covers N = 1
is_square_matrix_mutable(M) = mixed_array_trait(M) isa MatSqMut


############
# Conversion
############
"""
    convert_to_matrix_mixed(M::Matrix)

Matrix{T,2} -> nested mutable ArrayPartition (MatrixMixed)  
"""
function convert_to_matrix_mixed(A)
    if is_square_matrix_mutable(A)
        A
    else
        ArrayPartition(map(eachrow(A)) do rw
            ArrayPartition(map(rw) do el
                [el]
            end...)
        end...)::MatrixMixedCandidate
    end
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
function convert_to_array(A::ArrayPartition)
    m = length(A.x)
    n = length(A.x[1])
    if n != 1
        X = Array{Any, 2}(undef, m, n)
        for i = 1:m
            # Square. Can easily be dropped to extend functionality.
            @assert n == length(A.x[i]) 
            for j = 1:n
                X[i, j] = A.x[i][j]
            end
        end
        X
    else
        Vector{Any}(A)
    end
end

##########################
# IO nested ArrayPartition
##########################
# Decorated representation, trait based.
summary(io::IO, A::MatrixMixedCandidate) = summary(io, mixed_array_trait(A), A)
function summary(io::IO, ::MatSqMut, A)
    col = get(io, :unitsymbolcolor, :cyan)
    printstyled(io, color = col, "MatrixMixed as ")
    print(io, typeof(A))
    nothing
end
# Fallback, same as Base/show.jl:2803
summary(io::IO, ::NotMatSqMut, A::AbstractArray) = array_summary(io, A, axes(A))

# Un-decorated representation, trait-based
print(io::IO, A::MatrixMixedCandidate) =  print_as_MatrixCandidate(io, mixed_array_trait(A), A)
function print_as_MatrixCandidate(io::IO, ::MatSqMut, A::AbstractArray)
    X = convert_to_array(A)
    col = get(io, :unitsymbolcolor, :cyan)
    buf = IOBuffer()
    ioc = IOContext(buf, IOContext(io).dict)
    printstyled(ioc, color = col, "convert_to_matrix_mixed(")
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
# Fallback, same as Base/strings/io.jl:33
function print_as_MatrixCandidate(io::IO, ::NotMatSqMut, A::AbstractArray)
    lock(io)
    try
        show(io, A)
    finally
        unlock(io)
    end
    return nothing
end

# Human-readable text output, Julia parseable except coloured heading, trait-based.
Base.show(io::IO, m::MIME"text/plain", A::MatrixMixedCandidate) = Base.show(io, m, mixed_array_trait(A), A)
function Base.show(io::IO, ::MIME"text/plain", ::MatSqMut, A::MatrixMixedCandidate)
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

    # 2) update typeinfo
    #
    # it must come after printing the summary, which can exploit :typeinfo itself
    # (e.g. views)
    # we assume this function is always called from top-level, i.e. that it's not nested
    # within another "show" method; hence we always print the summary, without
    # checking for current :typeinfo (this could be changed in the future)
    io = IOContext(io, :typeinfo => eltype(X))

    # 2) show actual content
    recur_io = IOContext(io, :SHOWN_SET => X)
    Base.print_array(recur_io, X)
end
# Fallback, same as \RecursiveArrayTools\tU7uv\src\array_partition.jl:248
Base.show(io::IO, m::MIME"text/plain", ::NotMatSqMut, A::MatrixMixedCandidate) = show(io, m, A.x)

# Overloads RecursiveArrayTools.jl:26, which invokes showing this as 'Any'..., with too much header info:
Base.show(io::IO, A::MatrixMixedCandidate) = Base.show(io, mixed_array_trait(A), A)
function Base.show(io::IO, ::MatSqMut, A::MatrixMixedCandidate)
    # A normal matrix would have no type info here. We'll just provide a short summary with coloured 
    # highlighting to indicate that this is not a quite normal matrix.
    col = get(io, :unitsymbolcolor, :cyan)
    printstyled(io, color = col, "convert_to_matrix_mixed(")
    X = convert_to_array(A)
    show(io, X)
    printstyled(io, color = col, ")")
end
# Same as RecursiveArrayTools.jl:26
Base.show(io::IO, ::NotMatSqMut, A::MatrixMixedCandidate) = invoke(show, Tuple{typeof(io), Any}, io, A)

#####################################
# IO mutable ArrayPartion([x1]..[xN])
#####################################
# Decorated representation.
Base.summary(io::IO, ::RW(N)) where {N} = print(io, "$N-element mutable ArrayPartition")
#=
# Un-decorated representation
function Base.show(io::IO, v::RW(N)) where N
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

# Human-readable text output, Julia parseable except coloured heading.
function Base.show(io::IO, m::MIME"text/plain", X::RW(N)) where N
    # This is identical to the version for MatrixMixedCandidate above
    if isempty(X) && (get(io, :compact, false) || X isa Vector)
        return show(io, X)
    end
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

    # 2) update typeinfo
    #
    # it must come after printing the summary, which can exploit :typeinfo itself
    # (e.g. views)
    # we assume this function is always called from top-level, i.e. that it's not nested
    # within another "show" method; hence we always print the summary, without
    # checking for current :typeinfo (this could be changed in the future)
    io = IOContext(io, :typeinfo => eltype(X))

    # 2) show actual content
    recur_io = IOContext(io, :SHOWN_SET => X)
    Base.print_array(recur_io, X)
end

# Overloads RecursiveArrayTools.jl:26, which invokes showing this as 'Any'..., with too much header info:
Base.show(io::IO, A::MatrixMixedCandidate) = Base.show(io, mixed_array_trait(A), A)
function Base.show(io::IO, ::MatSqMut, A::MatrixMixedCandidate)
    # A normal matrix would have no type info here. We'll just provide a short summary with coloured 
    # highlighting to indicate that this is not a quite normal matrix.
    col = get(io, :unitsymbolcolor, :cyan)
    printstyled(io, color = col, "convert_to_matrix_mixed(")
    X = convert_to_array(A)
    show(io, X)
    printstyled(io, color = col, ")")
end
# Same as RecursiveArrayTools.jl:26
Base.show(io::IO, ::NotMatSqMut, A::MatrixMixedCandidate) = invoke(show, Tuple{typeof(io), Any}, io, A)

=#