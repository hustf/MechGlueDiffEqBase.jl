#####################
# Jacobian prototypes
#####################

"""
    jacobian_prototype_zero(x, vecfx)
-> type-stable, mutable Jacobian container prototype.
Assumes x is a vector of arguments to function f, and vecfx = f(x).
The Jacobian associates elements of x with columns, and
elements of vecfx with rows. If both vectors have elements with units,
each element in the Jacobian matrix could have unique units. This container
is intended to help the compiler making efficient machine code by foretelling
units, and can be used as if it was an ordinary Matrix{Any}.
"""
jacobian_prototype_zero(x, vecfx) = zero.(jacobian_prototype_nan(x, vecfx))




"""
    jacobian_prototype_nan(x, vecfx)
-> type-stable, mutable Jacobian container prototype.
Assumes x is a vector of arguments to function f, and vecfx = f(x).
The Jacobian associates elements of x with columns, and
elements of vecfx with rows. If both vectors have elements with units,
each element in the Jacobian matrix could have unique units. This container
is intended to help the compiler making efficient machine code by foretelling
units, and can be used as if it was an ordinary Matrix{Any}.
"""
function jacobian_prototype_nan(x::ArrayPartition, vecfx::ArrayPartition)
    # Immutable input
    typednan = (xel, fxel) -> [NaN * (fxel / xel)]
    genrow(fxel) = map(xel -> typednan(xel, fxel) , x)
    N = length(x)
    @assert N == length(vecfx)
    @assert x.x isa NTuple{N, Q}
    @assert vecfx.x isa NTuple{N, Q}
    map(genrow, vecfx)
end
function jacobian_prototype_nan(x::RW(N), vecfx::RW(N)) where N
    # Mutable input
    typednan = (xel, fxel) -> NaN * (fxel / xel)
    genrow(fxel) = map(xel -> typednan(xel, fxel) , x)
    Jint = map(genrow, vecfx)
    ArrayPartition(Jint...)
end
function jacobian_prototype_nan(x::RW(N), vecfx::ArrayPartition) where N
    # Mutable x, immutable f(x)
    typednan = (xel, fxel) -> NaN * (fxel / xel)
    genrow(fxel) = map(xel -> typednan(xel, fxel) , x)
    @assert length(vecfx) == N
    @assert vecfx.x isa NTuple{N, Q} "Inconsistent vecfx = f(x), must be mutable or not"
    map(genrow, vecfx)
end
function jacobian_prototype_nan(x::ArrayPartition, vecfx::RW(N)) where N
    # Immutable x, mutable f(x)
    typednan = (xel, fxel) -> [NaN * (fxel / xel)]
    genrow(fxel) = map(xel -> typednan(xel, fxel) , x)
    @assert length(x) == N
    @assert x.x isa NTuple{N, Q} "Inconsistent x.x, must be mutable or not"
    jprot = ArrayPartition(NTuple{N}(map(genrow, vecfx)))
    @debug "" jprot
    @assert genrow(vecfx[1]) isa RW(N)
    @assert genrow(vecfx[2]) isa RW(N)
    jprot
end
function jacobian_prototype_nan(x::E, vecfx::Vector{<:Q})
    N = length(vecfx)
    @assert length(x) == N
    # Immutable x, mutable vector f(x) with at least one element in x or vecfx with
    # physical dimension, none of which are ArrayPartitions.
    # Such differential equations are workable, easier to formulate, though perhaps
    # are not inferrable.
    jacobian_prototype_nan(ArrayPartition(x...), ArrayPartition(vecfx...))
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

    throw("Here we are! Or are we?")

    @debug "JacobianCache T"  T maxlog = 2
    @debug returntype
    @debug "Hei hei"
    fdtype isa Type && (fdtype = fdtype())
    inplace isa Type && (inplace = inplace())
    if fdtype == Val(:complex)
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
