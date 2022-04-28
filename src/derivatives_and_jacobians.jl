# The function signature in OrdinaryDiffEq.FiniteDiff is restrictive. "Real" excludes complex numbers,
# but that unfortunately excludes Quantity as well. A Quantity covering several types can be Real or Complex.
# compute_epsilon extends \FiniteDiff\src\epsilons.jl
@inline function compute_epsilon(::Val{:central}, x::T, relstep::Real, absstep::Quantity{T1, D, U}, dir=nothing) where {T<:Number, T1<:Real, D, U}
    max(relstep * abs(x), absstep)
end

@inline function compute_epsilon(::Val{:forward}, x::T, relstep::Real, absstep::Quantity{T1, D, U}, 
    dir = nothing) where {T<:Number, T1<:Real, D, U}
    max(relstep * oneunit(absstep), absstep)
end

@inline function compute_epsilon(::Val{:complex}, x::Quantity{T, D, U}, ::Union{Nothing,T1}=nothing, 
    ::Union{Nothing,Quantity{T, D, U}}=nothing, dir=nothing) where {T1<:Real, T<:Real, D, U}
    eps(Quantity{T, D, U})
end
 
# Extend Unitifu function for non-quantities
numtype(x::Type) = x

# From FiniteDiff\src\derivatives.jl:4. The only change here is the default absstep argument. 
function finite_difference_derivative(
    f,
    x::T,
    fdtype=Val(:central),
    returntype=eltype(x),
    f_x=nothing;
    relstep=default_relstep(fdtype, T),
    absstep=relstep * oneunit(x),
    dir=true) where {T<:Quantity}

    fdtype isa Type && (fdtype = fdtype())
    epsilon = compute_epsilon(fdtype, x, relstep, absstep, dir)
    if fdtype==Val(:forward)
        return (f(x+epsilon) - f(x)) / epsilon
    elseif fdtype==Val(:central)
        return (f(x+epsilon) - f(x-epsilon)) / (2*epsilon)
    elseif fdtype==Val(:complex) && returntype<:Real
        return imag(f(x+im*epsilon)) / epsilon
    end
    fdtype_error(returntype)
end

# We want to specialize on ArrayPartitions that 
# - can represent square matrices  
# - are mutable
# - have width and height > 1
# - are inferrable
# ...because Matrix{Any} is mostly not inferrable. For inferrability, we implement
# such matrix-like objects as nested ArrayPartition. 
# For mutability, the innermost type is a one-element vector.

# This imperfect definition of what we want to dispatch on includes immutable
# versions and empty tuples. It dispatches a bit too widely.
const MatrixCandidate = ArrayPartition{T, Tuple{U, V}} where {T, U<:ArrayPartition, V<:ArrayPartition}

# We use traits-based dispatch below:
abstract type MixedArray end
struct SqMatMut <: MixedArray end
struct NotSqMatMut <: MixedArray end
# The trait function returns a concrete type for the trait.
mixed_array_type(::T) where {T<:AbstractArray} = NotSqMatMut()             # Fallback
mixed_array_type(::ArrayPartition{Union{}, Tuple{}}) = NotSqMatMut()       # Covers N=0, see https://docs.julialang.org/en/v1/manual/methods/#Tuple-and-NTuple-arguments
mixed_array_type(::ArrayPartition{T, NTuple{N, U}}) where {N, T, V<:NTuple{N, Vector{<:T}}, U<:ArrayPartition{T, V}} = SqMatMut()
is_square_matrix_mutable(M) = mixed_array_type(M) isa SqMatMut




"""
    MatrixCandidate_arraypartition(A::Matrix)

Matrix{T,2} -> nested ArrayPartition (MatrixCandidate)  
"""
MatrixCandidate_arraypartition(A::Matrix) = ArrayPartition((ArrayPartition(rw...) for rw in eachrow(A))...)

"""
    similar_matrix(A::ArrayPartition)
Same as convert(Array{T, 2}, apa), but we avoid overloading base function on types 
defined by ArrayTools. 

This function should be avoided where speed matters.
"""
function similar_matrix(A::ArrayPartition)
    m = length(A.x)
    n = length(A.x[1])
    X = Array{Any, 2}(undef, m, n)
    for i = 1:m
        @assert n == length(A.x[i])
        for j = 1:n
            X[i, j] = A.x[i][j]
        end
    end
    X
end
"""
    row_vector(A::MatrixCandidate)

The container type we use has unclear distinction between row and column vector.
But the Jacobian for f: Rⁿ → R (the gradient) is a row vector, so call this 
instead of `similar_matrix` when the Jacobian is a gradient and you want clarity.
"""
function row_vector(A::ArrayPartition)
    @assert min(length(A.x), length(A.x[1])) == 1
    n = max(length(A.x), length(A.x[1]))
    X = Array{Any, 2}(undef, 1, n)
    for j = 1:n
        X[1, j] = A[j]
    end
    X
end
# Decorated representation of ArrayPartition "matrix"
function summary(io::IO, A::MatrixCandidate)
    col = get(io, :unitsymbolcolor, :cyan)
    printstyled(io, color = col, "Matrix representation of ")
    print(io, typeof(A))
    nothing
end

# Un-decorated representation of ArrayPartition "matrix"
print(io::IO, A::MatrixCandidate) =  print_as_MatrixCandidate(io, A)

function print_as_MatrixCandidate(io::IO, A::AbstractArray)
    X = similar_matrix(A)
    col = get(io, :unitsymbolcolor, :cyan)
    buf = IOBuffer()
    ioc = IOContext(buf, IOContext(io).dict)
    printstyled(ioc, color = col, "MatrixCandidate ArrayPartition:")
    prefix = String(take!(buf))
    isempty(X) ?
        Base._show_empty(io, X) :
        Base._show_nonempty(io, X, prefix)
end

function Base.show(io::IO, mime::MIME"text/plain", A::MatrixCandidate)
    # 0) show summary before setting :compact
    summary(io, A)
    isempty(A) && return
    print(io, ":")
    Base.show_circular(io, A) && return
    X = similar_matrix(A)
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

function Base.setindex(x::ArrayPartition{T}, v, i::Int) where T<:Quantity
    @assert min(length(x.x), length(x.x[1])) == 1
    map(enumerate(x)) do (j, x)
        j == i ? v : x
    end
end
"""
    jacobian_prototype_zero(x::ArrayPartition, vecfx::ArrayPartition)
Assume x is a vector of arguments to function f. Evaluate f with x outputs
vector fx. The Jacobian places associates elements of x with columns, and
elements of vecfx with rows. If both vectors have elements with units,
each element in the Jacobian matrix could have unique units. This container
is intended to help the compiler making efficient machine code by foretelling
units, and can be used as if it was an ordinary Matrix{Any}.
"""
function jacobian_prototype_zero(x::ArrayPartition, vecfx::ArrayPartition)
    typedzero = (xel, fxel) -> zero(fxel / xel)
    genrow(fxel) = map(xel -> typedzero(xel, fxel) , x)
    map(genrow, vecfx)
end

function jacobian_prototype_nan(x::ArrayPartition, vecfx::ArrayPartition)
    typednan = (xel, fxel) -> NaN * (fxel / xel)
    genrow(fxel) = map(xel -> typednan(xel, fxel) , x)
    jprot = map(genrow, vecfx)
    @info "x" x maxlog=2
    @info "vecfx" vecfx maxlog=2
    @info "jprot = $jprot" maxlog=2
    @info "MatrixCandidate" jprot isa MatrixCandidate vecfx
    jprot
end
function alloc_DF(x::ArrayPartition{<:AbstractQuantity, <:Tuple} , F)
    @info "All is good" maxlog=2
    @info "x = " x maxlog = 2
    jprot = jacobian_prototype_nan(x, F)
    @info "x" x maxlog=2
    @info "F" F maxlog=2
    @info "jprot = $jprot" maxlog=2
    @info "MatrixCandidate" jprot isa MatrixCandidate maxlog=2
    jprot
end


# From FiniteDiff\src\jacobians.jl:133
# Dispatches on ArrayPartition, the only change is
# default absstep, which needs to conform to the units of x.
function finite_difference_jacobian(f, x::ArrayPartition,
        fdtype::Val     = Val(:forward), 
        returntype = fdtype == Val(:complex) ? numtype(eltype(f(x))) : eltype(f(x)),
        f_in       = nothing;
        relstep=default_relstep(fdtype, eltype(x)),
        absstep = relstep .* oneunit.(x), # Units can vary between 'rows' in the ArrayPartition.
        colorvec = 1:length(x),
        sparsity = nothing,
        jac_prototype = nothing,
        dir=true)
    if f_in isa Nothing
        fx = f(x)
    else
        @debug "Untested" maxlog = 1
        fx = f_in
    end
    if fdtype == Val(:complex)
        xcomp = complex.(zero(x))
        cache = JacobianCache(xcomp, f(xcomp), fdtype, returntype)
    else
        cache = JacobianCache(x, fx, fdtype, returntype)
    end
    # Targeting method extending FiniteDiff\src\jacobians.jl:155,
    # defined below.
    finite_difference_jacobian(f, x, cache, f_in;
        relstep, absstep, colorvec, sparsity, jac_prototype, dir)
end

function finite_difference_jacobian(
        f,
        x::ArrayPartition,
        cache::JacobianCache{T1,T2,T3,cType,sType,fdtype,returntype},
        f_in=nothing;
        relstep=default_relstep(fdtype, eltype(x)),
        absstep=relstep,
        colorvec = cache.colorvec,
        sparsity = cache.sparsity,
        jac_prototype = nothing,
        dir=true) where {T1,T2,T3,cType,sType,fdtype,returntype}

    x1, fx, fx1 = cache.x1, cache.fx, cache.fx1

    if !(f_in isa Nothing)
        @debug "Untested" maxlog = 1
        vecfx = MechGlueDiffEqBase._vec(f_in)
    elseif fdtype == Val(:forward)
        vecfx = MechGlueDiffEqBase._vec(f(x))
    elseif fdtype == Val(:complex) && returntype <: Real
        vecfx = real(fx)
    else
        vecfx = MechGlueDiffEqBase._vec(fx)
    end
    vecx = MechGlueDiffEqBase._vec(x)
    vecx1 = MechGlueDiffEqBase._vec(x1)

    J = jac_prototype isa Nothing ? (sparsity isa Nothing ? jacobian_prototype_zero(x, vecfx) : zeros(eltype(x),size(sparsity))) : zero(jac_prototype)
    nrows = length(J.x)
    ncols = length(J.x[1])
   
    if !(sparsity isa Nothing)
        @debug "Untested" maxlog = 1
        rows_index, cols_index = ArrayInterface.findstructralnz(sparsity)
        rows_index = [rows_index[i] for i in 1:length(rows_index)]
        cols_index = [cols_index[i] for i in 1:length(cols_index)]
    end

    if fdtype == Val(:forward)
        function calculate_Ji_forward(i)
            x_save = ArrayInterface.allowed_getindex(vecx, i)
            epsilon = compute_epsilon(Val(:forward), x_save, relstep, absstep[i], dir)
            su = x_save + epsilon
            _vecx1 = Base.setindex(vecx, su, i)
            _x1 = reshape(_vecx1, axes(x))
            vecfx1 = MechGlueDiffEqBase._vec(f(_x1))
            (vecfx1-vecfx) / epsilon
        end
        if jac_prototype isa Nothing && sparsity isa Nothing
            Jvec = map(calculate_Ji_forward, 1:maximum(colorvec))
            J = ArrayPartition(map(ArrayPartition, zip(Jvec...))...)
        else
            @debug "Untested" maxlog = 1
            @inbounds for color_i ∈ 1:maximum(colorvec)
                if sparsity isa Nothing
                    dx = calculate_Ji_forward(color_i)
                    J = J + _make_Ji(J, eltype(x), dx, color_i, nrows, ncols)
                else
                    tmp = norm(vecx .* (colorvec .== color_i))
                    epsilon = compute_epsilon(Val(:forward), sqrt(tmp), relstep, absstep, dir)
                    _vecx = @. vecx + epsilon * (colorvec == color_i)
                    _x = reshape(_vecx, axes(x))
                    vecfx1 = _vec(f(_x))
                    dx = (vecfx1-vecfx)/epsilon
                    Ji = _make_Ji(J,rows_index,cols_index,dx,colorvec,color_i,nrows,ncols)
                    J = J + Ji
                end
            end
        end
    elseif fdtype == Val(:central)
        function calculate_Ji_central(i)
            x1_save = ArrayInterface.allowed_getindex(vecx1,i)
            x_save = ArrayInterface.allowed_getindex(vecx,i)
            epsilon = compute_epsilon(Val(:central), x_save, relstep, absstep[i], dir)
            _vecx1 = Base.setindex(vecx1,x1_save+epsilon,i)
            _vecx = Base.setindex(vecx,x_save-epsilon,i)
            _x1 = reshape(_vecx1, axes(x))
            _x = reshape(_vecx, axes(x))
            vecfx1 = _vec(f(_x1))
            vecfx = _vec(f(_x))
            dx = (vecfx1-vecfx)/(2epsilon)
            return dx
        end

        if jac_prototype isa Nothing && sparsity isa Nothing
            Jvec = map(calculate_Ji_central, 1:maximum(colorvec))
            J = ArrayPartition(map(ArrayPartition, zip(Jvec...))...)
        else
            @debug "Untested" maxlog = 1
            @inbounds for color_i ∈ 1:maximum(colorvec)
                if sparsity isa Nothing
                    @debug "Untested" maxlog = 1
                    dx = calculate_Ji_central(color_i)
                    J = J + _make_Ji(J, eltype(x), dx, color_i, nrows, ncols)
                else
                    @debug "Untested" maxlog = 1
                    tmp = norm(vecx1 .* (colorvec .== color_i))
                    epsilon = compute_epsilon(Val(:forward), sqrt(tmp), relstep, absstep, dir)
                    _vecx1 = @. vecx1 + epsilon * (colorvec == color_i)
                    _vecx = @. vecx - epsilon * (colorvec == color_i)
                    _x1 = reshape(_vecx1, axes(x))
                    _x = reshape(_vecx, axes(x))
                    vecfx1 = _vec(f(_x1))
                    vecfx = _vec(f(_x))
                    dx = (vecfx1-vecfx)/(2epsilon)
                    Ji = _make_Ji(J,rows_index,cols_index,dx,colorvec,color_i,nrows,ncols)
                    J = J + Ji

                end
            end
        end
    elseif fdtype == Val(:complex) && returntype <: Real
        function calculate_Ji_complex(i)
            x_save = ArrayInterface.allowed_getindex(vecx,i)
            epsilon = compute_epsilon(Val(:complex), x_save, relstep, absstep[i], dir)
            _vecx = Base.setindex(complex.(vecx), x_save + im * oneunit(x_save) * epsilon, i)
            _x = reshape(_vecx, axes(x))
            vecfx = MechGlueDiffEqBase._vec(f(_x))
            imag(vecfx) / (epsilon * oneunit(x_save))
        end

        if jac_prototype isa Nothing && sparsity isa Nothing
            Jvec = map(calculate_Ji_complex, 1:maximum(colorvec))
            J = ArrayPartition(map(ArrayPartition, zip(Jvec...))...)
        else
            @debug "Untested" maxlog = 1
            @inbounds for color_i ∈ 1:maximum(colorvec)
                if sparsity isa Nothing
                    dx = calculate_Ji_complex(color_i)
                    J = J + _make_Ji(J, eltype(x), dx, color_i, nrows, ncols)
                else
                    @debug "Untested" maxlog = 1
                    _vecx = @. vecx + im * epsilon * (colorvec == color_i)
                    _x = reshape(_vecx, axes(x))
                    vecfx = _vec(f(_x))
                    dx = imag(vecfx)/epsilon
                    Ji = _make_Ji(J,rows_index,cols_index,dx,colorvec,color_i,nrows,ncols)
                    J = J + Ji
                end
            end
        end
    else
        @debug "Untested" maxlog = 1
        fdtype_error(returntype)
    end
    J
end

# We avoid overloading type generators as far as possible,
# but this seems necessary in order to target 
# NLSolverBase/src/oncedifferentiable.jl/OnceDifferentiable:94
# j_finitediff_cache = FiniteDiff.JacobianCache(copy(x_seed), copy(F), copy(F), fdtype)
# Extending FiniteDiff/src/jacobians.jl:68, the difference is 
function JacobianCache(
    x1::ArrayPartition{T},
    fx::ArrayPartition{<:AbstractQuantity, Tuple} ,
    fx1::ArrayPartition{<:AbstractQuantity, Tuple}, # TODO test joining types fx, fx1
    fdtype     :: Union{Val{T1},Type{T1}} = Val(:forward),
    returntype :: Type{T2} = fdtype == Val(:complex) ? numtype(eltype(fx)) : eltype(fx);
    inplace    :: Union{Val{T3},Type{T3}} = Val(true),
    colorvec = 1:length(x1),
    sparsity = nothing) where {T<:AbstractQuantity, T1, T2, T3}

    @info T
    @info returntype
    @info "Hei hei"
    fdtype isa Type && (fdtype = fdtype())
    inplace isa Type && (inplace = inplace())
    if fdtype==Val(:complex)
        !(returntype<:Real) && fdtype_error(returntype)

        if eltype(fx) <: Real
            _fx  = false .* im .* fx
        else
            _fx = fx
        end
        if eltype(x1) <: Real
            _x1  = false .* im .* x1
        else
            _x1 = x1
        end
    else
        _x1 = x1
        @assert eltype(fx) == T2
        @assert eltype(fx1) == T2
        _fx = fx
    end
    JacobianCache{typeof(_x1),typeof(_fx),typeof(fx1),typeof(colorvec),typeof(sparsity),fdtype,returntype}(_x1,_fx,fx1,colorvec,sparsity)
end
