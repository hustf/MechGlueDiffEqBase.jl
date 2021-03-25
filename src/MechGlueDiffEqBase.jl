module MechGlueDiffEqBase

using MechanicalUnits
import MechanicalUnits.Unitfu as Unitful
import DiffEqBase
import DiffEqBase: value, ODE_DEFAULT_NORM, UNITLESS_ABS2

function __init__()
    # Glue code copied from DiffEqBase.jl init.jl
    # Here, we are using an alias. We are actually using types from Unitfu, but
    # with the alias Unitful. Unitfu should only vary from Unitful in how it shows and parses quantities.
    value(x::Type{Unitful.AbstractQuantity{T,D,U}}) where {T,D,U} = T
    value(x::Unitful.AbstractQuantity) = x.val
    @inline function ODE_DEFAULT_NORM(u::AbstractArray{<:Unitful.AbstractQuantity,N},t) where {N}
        sqrt(sum(x->ODE_DEFAULT_NORM(x[1],x[2]),zip((value(x) for x in u),Iterators.repeated(t))) / length(u))
    end
    @inline function ODE_DEFAULT_NORM(u::Array{<:Unitful.AbstractQuantity,N},t) where {N}
        sqrt(sum(x->ODE_DEFAULT_NORM(x[1],x[2]),zip((value(x) for x in u),Iterators.repeated(t))) / length(u))
    end
    @inline ODE_DEFAULT_NORM(u::Unitful.AbstractQuantity,t) = abs(value(u))
    @inline UNITLESS_ABS2(x::Unitful.AbstractQuantity) = real(abs2(x)/oneunit(x)*oneunit(x))
end


end
