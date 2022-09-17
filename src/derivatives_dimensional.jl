###############################
# Differentiation of quantities
###############################

# The function signature in OrdinaryDiffEq.FiniteDiff is restrictive. "Real" excludes complex numbers,
# but that unfortunately excludes Quantity as well. A Quantity covering several types can be Real or Complex.
# compute_epsilon extends \FiniteDiff\src\epsilons.jl
@inline function compute_epsilon(::Val{:central}, x::T, relstep::Real, absstep::Quantity{T1, D, U}, dir = nothing) where {T<:Number, T1<:Real, D, U}
    @debug "compute_epsilon:9 central" T x relstep absstep maxlog = 2
    max(relstep * abs(x), absstep)
end

@inline function compute_epsilon(::Val{:central}, x::Quantity{T1, D, U}, relstep::Real, absstep::Real, dir = nothing) where {T1<:Real, D, U}
    @debug "compute_epsilon:14 central quantity" T1 x relstep absstep maxlog = 2
    return max(relstep*abs(x), absstep * oneunit(x))
end

@inline function compute_epsilon(::Val{:forward}, x::T, relstep::Real, absstep::Quantity{T1, D, U},
    dir = nothing) where {T<:Number, T1<:Real, D, U}
    @debug "compute_epsilon:20 forward quantity" T1 x relstep absstep maxlog = 2
    max(relstep * oneunit(absstep), absstep)
end

@inline function compute_epsilon(::Val{:forward}, x::Quantity{T1, D, U}, relstep::Real, absstep::Real, dir = nothing) where {T1<:Real, D, U}
    @debug "compute_epsilon:25 forward quantity" T1 x relstep absstep maxlog = 2
    throw("unused, untested")
    return max(relstep*abs(x), absstep)
end

@inline function compute_epsilon(::Val{:complex}, x::Quantity{T, D, U}, ::Union{Nothing,T1} = nothing,
    ::Union{Nothing,Quantity{T, D, U}} = nothing, dir = nothing) where {T1<:Real, T<:Real, D, U}
    @debug "compute_epsilon:32 complex quantity" T x relstep absstep maxlog = 2
    throw("unused, untested")
    eps(Quantity{T, D, U})
end

@inline function compute_epsilon(::Val{:complex}, x::Quantity{T1, D, U}, relstep::Real, absstep::Real, dir = nothing) where {T1<:Real, D, U}
    @debug "compute_epsilon complex quantity" x relstep absstep maxlog = 2
    throw("unused, untested")
    return eps(Quantity{T, D, U})
end


##############################
# (Updating) Jacobian matrices
# with mixed element types
##############################

# From FiniteDiff\src\derivatives.jl:4. The only change here is the default absstep argument.
function finite_difference_derivative(
    f,
    x::T,
    fdtype = Val(:central),
    returntype = eltype(x),
    f_x = nothing;
    relstep = default_relstep(fdtype, T),
    absstep = relstep * oneunit(x),
    dir = true) where {T<:Quantity}

    fdtype isa Type && (fdtype = fdtype())
    epsilon = compute_epsilon(fdtype, x, relstep, absstep, dir)
    if fdtype == Val(:forward)
        return (f(x+epsilon) - f(x)) / epsilon
    elseif fdtype == Val(:central)
        return (f(x+epsilon) - f(x-epsilon)) / (2*epsilon)
    elseif fdtype == Val(:complex) && returntype<:Real
        return imag(f(x+im*epsilon)) / epsilon
    end
    fdtype_error(returntype)
end


