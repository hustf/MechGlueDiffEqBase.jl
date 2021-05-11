module MechGlueDiffEqBase
import Base: similar
import Unitfu: AbstractQuantity, Quantity, ustrip, norm, unit, zero
import Unitfu: Dimensions, FreeUnits
import DiffEqBase: value, ODE_DEFAULT_NORM, UNITLESS_ABS2
import DiffEqBase: calculate_residuals, @muladd
using RecursiveArrayTools
export value, ODE_DEFAULT_NORM, UNITLESS_ABS2, Unitfu, AbstractQuantity, Quantity
export norm , ArrayPartition, similar, zero

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

@inline function UNITLESS_ABS2(u::AbstractArray{<:AbstractQuantity,N} where N)
    map(UNITLESS_ABS2, u)
end
@inline function UNITLESS_ABS2(u::AbstractArray{Quantity{T},N}) where {N, T}
    map(UNITLESS_ABS2, u)
end

@inline function UNITLESS_ABS2(x::T) where T <: AbstractQuantity
    xul = x / oneunit(T)
    abs2(xul)
end
@inline function UNITLESS_ABS2(x::Quantity{T, D, U}) where {T, D, U}
    xul = x / oneunit(Quantity{T, D, U})
    abs2(xul)::T
end

# Vectors with compatible units, treat as normal
zero(x::Vector{Quantity{T, D, U}}) where {T,D,U} = fill!(similar(x), zero(Quantity{T, D, U}))
# Vectors with incompatible units, special inferreable treatment
function zero(x::Vector{Q}) where {Q<:AbstractQuantity{T, D, U} where {D,U}} where T
    x0 = copy(x)
    for i in eachindex(x0)
        x = x0[i]
        x0[i] = 0 * x * sign(x)
    end
    x0
end

# Vectors with compatible units, treat as normal
similar(x::Vector{Quantity{T, D, U}}) where {T,D,U} = Vector{Quantity{T, D, U}}(undef, size(x,1))
#similar(a::Array{T,1}) where {T}                    = Vector{T}(undef, size(a,1))
# Vectors with incompatible units, special inferreable treatment
# VERY similar is still similar and (very slightly) different
similar(x::Vector{Q}) where {Q<:AbstractQuantity{T, D, U} where {D,U}} where T = copy(x)




# KISS pre-compillation to reduce loading times
# This is simply a boiled-down obfuscated test_4.jl
import Unitfu: m, s, kg, N, ∙
let
    r0ul = [1131.340, -2282.343, 6672.423]
    r0ul = [1131.340, -2282.343, 6672.423]
    v0ul = [-5.64305, 4.30333, 2.42879]
    rv0ul = ArrayPartition(r0ul,v0ul)
    ODE_DEFAULT_NORM(rv0ul, 0.0)
    r0 = [1131.340, -2282.343, 6672.423]∙kg
    v0 = [-5.64305, 4.30333, 2.42879]∙kg/s
    rv0 = ArrayPartition(r0, v0)
    ODE_DEFAULT_NORM(rv0, 0.0s)
    r0 = [1.0kg, 2.0N, 3.0m/s, 4.0m/s]
    v0 = [1.0kg/s, 2.0N/s, 3.0m/s^2, 4m/s^2]
    rv0 = ArrayPartition(r0, v0)
    r0ul = [1131.340, -2282.343, 6672.423]
    v0ul = [-5.64305, 4.30333, 2.42879]
    rv0ul = ArrayPartition(r0ul, v0ul)
    r0 = [1131.340, -2282.343, 6672.423]∙kg
    r1 = [1kg, 2.0m]
    zero(r0)
    zero(r1)
    rv0 = ArrayPartition(r0)
    zero(rv0)
    rv1 = ArrayPartition(r1)
    zero(rv1)
    r0 = [1.0kg, -2kg, 3m/s, 4m/s]
    zero(r0)
    rv0 = ArrayPartition(r0)
    zer = zero(rv0)
    zer == [0.0kg, 0.0kg, 0.0m/s, 0.0m/s]
    r0 = [1131.340, -2282.343, 6672.423]∙kg
    simi = similar(r0)
    rv0 = ArrayPartition(r0)
    sima = similar(rv0)
    r0 = [1.0kg, -2kg, 3m/s, 4m/s]
    similar(r0)
    rv0 = ArrayPartition(r0)
    sima = similar(rv0)
    r0ul = [1131.340, -2282.343, 6672.423]
    v0ul = [-5.64305, 4.30333, 2.42879]
    rv0ul = ArrayPartition(r0ul,v0ul)
    UNITLESS_ABS2(rv0ul)
    r0 = [1.0kg, 2.0N, 3.0m/s, 4.0m/s]
    v0 = [1.0kg/s, 2.0N/s, 3.0m/s^2, 4m/s^2]
    rv0 = ArrayPartition(r0, v0)
    UNITLESS_ABS2(1.0kg)
    UNITLESS_ABS2(r0)
    UNITLESS_ABS2(rv0)
    nothing
end

end
