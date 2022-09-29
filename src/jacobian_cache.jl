##############################################
# Jacobian cache outside constructors.
# 1) The change here is 
#     eltype(..) <: Real 
#   -> numtype(eltype(..)) <: Real
# Quantity{Real or Complex} is not <:Real
#
# 2) Traits-based dispatch is used.
#
# 3) These types do not yet implement sparsity,
#    so a barrier @assert is included here.
##############################################

# One (+) argument
function JacobianCache(
    x::MixedCandidate,
    fdtype     :: Union{Val{T1},Type{T1}} = Val(:forward),
    returntype :: Type{T2} = eltype(x);
    inplace    :: Union{Val{T3},Type{T3}} = Val(true),
    colorvec = 1:length(x),
    sparsity = nothing) where {T1,T2,T3}
    #
    JacobianCache(mixed_array_trait(x), x, fdtype, returntype; inplace, colorvec, sparsity)
end
function JacobianCache(
    ::VecMut,
    x,
    fdtype     :: Union{Val{T1},Type{T1}} = Val(:forward),
    returntype :: Type{T2} = eltype(x);
    inplace    :: Union{Val{T3},Type{T3}} = Val(true),
    colorvec = 1:length(x),
    sparsity = nothing) where {T1,T2,T3}
    #
    @assert sparsity isa Nothing
    fdtype isa Type && (fdtype = fdtype())
    inplace isa Type && (inplace = inplace())
    if numtype(eltype(x)) <: Real && fdtype==Val(:complex)
        x1  = false .* im .* x
        _fx = false .* im .* x
    else
        x1 = copy(x)
        _fx = copy(x)
    end

    if fdtype==Val(:complex)
        _fx1  = nothing
    else
        _fx1 = copy(x)
    end

    JacobianCache(x1,_fx,_fx1,fdtype,returntype;colorvec,sparsity)
end

# Two (+) argument
function JacobianCache(
    x::MixedCandidate,
    fx,
    fdtype     :: Union{Val{T1},Type{T1}} = Val(:forward),
    returntype :: Type{T2} = eltype(x);
    inplace    :: Union{Val{T3},Type{T3}} = Val(true),
    colorvec = 1:length(x),
    sparsity = nothing) where {T1,T2,T3}
    #
    JacobianCache(mixed_array_trait(x), x, fx, fdtype, returntype; inplace, colorvec, sparsity)
end
function JacobianCache(
    ::VecMut,
    x ,
    fx,
    fdtype     :: Union{Val{T1},Type{T1}} = Val(:forward),
    returntype :: Type{T2} = eltype(x);
    inplace    :: Union{Val{T3},Type{T3}} = Val(true),
    colorvec = 1:length(x),
    sparsity = nothing) where {T1,T2,T3}
    #
    @assert sparsity isa Nothing
    fdtype isa Type && (fdtype = fdtype())
    inplace isa Type && (inplace = inplace())
    if numtype(eltype(x)) <: Real && fdtype==Val(:complex)
        x1  = false .* im .* x
    else
        x1 = copy(x)
    end

    if numtype(eltype(fx)) <: Real && fdtype==Val(:complex)
        _fx = false .* im .* fx
    else
        _fx = copy(fx)
    end

    if fdtype==Val(:complex)
        _fx1  = nothing
    else
        _fx1 = copy(fx)
    end

    JacobianCache(x1,_fx,_fx1,fdtype,returntype;colorvec,sparsity)
end

# Three (+) argument
function JacobianCache(
    x1::MixedCandidate,
    fx,
    fx1,
    fdtype     :: Union{Val{T1},Type{T1}} = Val(:forward),
    returntype :: Type{T2} = eltype(fx);
    inplace    :: Union{Val{T3},Type{T3}} = Val(true),
    colorvec = 1:length(x1),
    sparsity = nothing) where {T1,T2,T3}
    #
    JacobianCache(mixed_array_trait(x1), x1, fx, fx1, fdtype, returntype; inplace, colorvec, sparsity)
end

function JacobianCache(
    ::VecMut,
    x1 ,
    fx ,
    fx1,
    fdtype     :: Union{Val{T1},Type{T1}} = Val(:forward),
    returntype :: Type{T2} = eltype(fx);
    inplace    :: Union{Val{T3},Type{T3}} = Val(true),
    colorvec = 1:length(x1),
    sparsity = nothing) where {T1,T2,T3}

    @assert sparsity isa Nothing

    fdtype isa Type && (fdtype = fdtype())
    inplace isa Type && (inplace = inplace())
    if fdtype==Val(:complex)
        !(numtype(returntype) <: Real) && fdtype_error(returntype)

        if numtype(eltype(fx)) <: Real
            _fx  = false .* im .* fx
        else
            _fx = fx
        end
        if numtype(eltype(x1)) <: Real
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
    _x2 = zero(_x1)
    JacobianCache{typeof(_x1),typeof(_x2),typeof(_fx),typeof(fx1),typeof(colorvec),typeof(sparsity),fdtype,returntype}(_x1,_x2,_fx,fx1,colorvec,sparsity)
end
