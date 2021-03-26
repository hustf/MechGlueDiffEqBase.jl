module MechGlueDiffEqBase
import MechanicalUnits.Unitfu.AbstractQuantity
import DiffEqBase: value, ODE_DEFAULT_NORM, UNITLESS_ABS2
export value, ODE_DEFAULT_NORM, UNITLESS_ABS2, Unitfu

function value(x::Type{AbstractQuantity{T,D,U}}) where {T,D,U}
    T
end
function value(x::AbstractQuantity)
    x.val
end
@inline function ODE_DEFAULT_NORM(u::AbstractArray{<:AbstractQuantity,N},t) where {N}
    sqrt(sum(x->ODE_DEFAULT_NORM(x[1],x[2]),zip((value(x) for x in u),Iterators.repeated(t))) / length(u))
end
@inline function ODE_DEFAULT_NORM(u::Array{<:AbstractQuantity,N},t) where {N}
    sqrt(sum(x->ODE_DEFAULT_NORM(x[1],x[2]),zip((value(x) for x in u),Iterators.repeated(t))) / length(u))
end
@inline function ODE_DEFAULT_NORM(u::AbstractQuantity,t)
    abs(value(u))
end
@inline function UNITLESS_ABS2(x::AbstractQuantity)
    real(abs2(x)/oneunit(x)*oneunit(x))
end


end
