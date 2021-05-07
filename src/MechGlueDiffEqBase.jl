module MechGlueDiffEqBase
import Unitfu: AbstractQuantity, Quantity, ustrip, norm
import DiffEqBase: value, ODE_DEFAULT_NORM, UNITLESS_ABS2, zero
import DiffEqBase: calculate_residuals, @muladd
using RecursiveArrayTools
export value, ODE_DEFAULT_NORM, UNITLESS_ABS2, Unitfu, AbstractQuantity, Quantity
export norm, ArrayPartition

Base.zero(A::ArrayPartition{<:AbstractQuantity{T},S}) where {T<:Number,S} = zero.(A)

# This is identical to what DiffEqBase defines for Unitful
function value(x::Type{AbstractQuantity{T,D,U}}) where {T,D,U}
    T
end
# This is different from what DiffEqBase defines for Unitful
value(::Type{<:AbstractQuantity{T,D,U}}) where {T,D,U<:Core.TypeofBottom} = Base.undef_ref_str
value(x::Q) where {Q<:AbstractQuantity} = ustrip(x)


# This is identical to what DiffEqBase defines for Unitful
@inline function ODE_DEFAULT_NORM(u::AbstractArray{<:AbstractQuantity,N},t) where {N}
    # Support adaptive errors should be errorless for exponentiation
    sqrt(sum(x->ODE_DEFAULT_NORM(x[1],x[2]),zip((value(x) for x in u),Iterators.repeated(t))) / length(u))
end
# This is identical to what DiffEqBase defines for Unitful
@inline function ODE_DEFAULT_NORM(u::Array{<:AbstractQuantity,N},t) where {N}
    sqrt(sum(x->ODE_DEFAULT_NORM(x[1],x[2]),zip((value(x) for x in u),Iterators.repeated(t))) / length(u))
end

# This is identical to what DiffEqBase defines for Unitful

@inline function ODE_DEFAULT_NORM(u::AbstractQuantity, t)
    abs(ustrip(u))
end
# This is slightly different from what  DiffEqBase defines for Unitful
@inline function UNITLESS_ABS2(x::AbstractQuantity)
    real(abs2(x) / (oneunit(x)^2))
end
end
