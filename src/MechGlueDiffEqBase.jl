module MechGlueDiffEqBase
# TODO: Don't import. Use!
import Base: similar, getindex, setindex!, inv, +, -, zip
import Unitfu: AbstractQuantity, Quantity, ustrip, norm, unit, zero, numtype
import Unitfu: uconvert, dimension, ∙
import Unitfu: Dimensions, Dimension, FreeUnits, NoUnits, DimensionlessQuantity
import DiffEqBase: value, ODE_DEFAULT_NORM, UNITLESS_ABS2, remake, abs2_and_sum
import DiffEqBase: calculate_residuals, @muladd, __solve, BVProblem, solve
import BoundaryValueDiffEq
using BoundaryValueDiffEq: Shooting
using RecursiveArrayTools
import RecursiveArrayTools.unpack
using RecursiveArrayTools: ArrayPartitionStyle, npartitions, unpack_args
import OrdinaryDiffEq
import OrdinaryDiffEq.FiniteDiff: compute_epsilon, finite_difference_derivative
import OrdinaryDiffEq.FiniteDiff: finite_difference_jacobian, finite_difference_jacobian!, JacobianCache
using OrdinaryDiffEq.FiniteDiff: default_relstep, fdtype_error
import Base: show, summary, print, setindex, size, ndims
import Base: \, IndexStyle, axes, BroadcastStyle, Broadcast.combine_styles, copy
using Base: array_summary, throw_boundserror, broadcasted
using OrdinaryDiffEq.FiniteDiff: _vec
using Printf
import ArrayInterfaceCore
import NLSolversBase
import NLSolversBase: alloc_DF, OnceDifferentiable, NonDifferentiable, x_of_nans
using NLSolversBase: f!_from_f, df!_from_df, fdf!_from_fdf, value_jacobian!!, AbstractObjective
import NLsolve
import NLsolve: nlsolve, trust_region, assess_convergence, check_isfinite, converged
using NLsolve: NewtonTrustRegionCache, dogleg!
import LinearAlgebra
using LinearAlgebra: require_one_based_indexing, istril, istriu, lu, wrapperop, MulAddMul
import LinearAlgebra: mul!, *
using LinearAlgebra: Diagonal, LowerTriangular, UpperTriangular, AdjOrTransAbsVec, Transpose
using LinearAlgebra: generic_matvecmul!
export value, ODE_DEFAULT_NORM, UNITLESS_ABS2, Unitfu, AbstractQuantity, Quantity, FreeUnits
export @import_expand, ∙
export norm , ArrayPartition, similar, zero, compute_epsilon
export jacobian_prototype_zero, jacobian_prototype_nan
export finite_difference_derivative, finite_difference_jacobian, show, summary, print
export MixedCandidate, convert_to_array, JacobianCache
export numtype, alloc_DF, mixed_array_trait, convert_to_mixed
export MatSqMut, VecMut, NotMixed, NotMixed, Single, Empty
export is_square_matrix_mutable, is_vector_mutable_stable
export OnceDifferentiable, DIMENSIONAL_NLSOLVE, check_isfinite


# TODO: wash list, 'using' over 'import'

# recursive types of ArrayPartition
include("io_traits_conversion.jl")

# Extended imported functions from base are not currently exported.

@inline UNITLESS_ABS2(x::Quantity)  = abs2(ustrip(x))
@inline UNITLESS_ABS2(u::MixedCandidate) = UNITLESS_ABS2(mixed_array_trait(u), u)
@inline function UNITLESS_ABS2(::VecMut, u::ArrayPartition{<:Quantity{T}}) where T
    mapreduce(v -> UNITLESS_ABS2(first(v)), + , u.x; init = zero(T))
end
@inline function UNITLESS_ABS2(::MatSqMut, u::ArrayPartition{<:Quantity{T}}) where T
    acc = zero(T)
    for rw in u.x
        acc  += UNITLESS_ABS2(VecMut(), rw)
    end
    acc
end
@inline function UNITLESS_ABS2(::VecMut, u::ArrayPartition{T}) where T <: Real
    mapreduce(v -> UNITLESS_ABS2(first(v)), + , u.x, init = zero(T))
end
@inline function UNITLESS_ABS2(::MatSqMut, u::ArrayPartition{T}) where T <: Real
    acc = zero(T)
    for rw in u.x
        acc  += UNITLESS_ABS2(VecMut(), rw)
    end
    acc
end


# This is identical to what DiffEqBase defines for Unitful
value(x::Type{Quantity{T,D,U}}) where {T,D,U} = T

# This is different from what DiffEqBase defines for Unitful
value(::Type{<:AbstractQuantity{T,D,U}}) where {T,D,U<:Core.TypeofBottom} = Base.undef_ref_str
value(x::Quantity) = ustrip(x)

# This is identical to what DiffEqBase defines for Unitful.
@inline ODE_DEFAULT_NORM(u::Quantity, t) = abs(ustrip(u))

# For types of ArrayPartition defined in io_traits_conversion.jl
@inline ODE_DEFAULT_NORM(u::MixedCandidate, t) = ODE_DEFAULT_NORM(mixed_array_trait(u), u, t)
@inline function ODE_DEFAULT_NORM(::UnionVecSqMut, u, t)
    un = ustrip.(u)
    y = sum(abs2, un; init = zero(eltype(un)))
    sqrt(real(y) / length(un))
end


# Vectors with compatible units, treat as normal
similar(x::Vector{Quantity{T, D, U}}) where {T,D,U} = Vector{Quantity{T, D, U}}(undef, size(x,1))
# Vectors with incompatible units, special inferreable treatment
# VERY similar is still similar and (very slightly) different
similar(x::Vector{Q}) where {Q<:AbstractQuantity{T, D, U} where {D,U}} where T = copy(x)


# Vectors with compatible units, treat as normal
similar(x::Vector{Quantity{T, D, U}}, S::Type) where {T,D,U} = Vector{S}(undef, size(x,1))
# Vectors with incompatible units, special inferreable treatment
function similar(x::Vector{Q}, S::Type) where {Q<:AbstractQuantity{T, D, U} where {D, U}} where T
    x0 = Vector{S}(undef, size(x, 1))
end


include("broadcast_mixed_matrix.jl")
include("derivatives_dimensional.jl")
include("jacobian_prototypes.jl")
include("once_differentiable.jl")
include("multiply_divide.jl")
include("solve_dimensional.jl")
include("trustregion_dimensional.jl")
include("jacobians_dimensional.jl")
include("utils_dimensional.jl")


# KISS pre-compillation to reduce loading times
# This is simply a boiled-down obfuscated test_013.jl
# It is commented out because there is no apparent effect.
#=
import Unitfu: m, km, s, kg, inch, °, ∙
    # Constants we don't think we'll change ever
    x₀ = 0.0km
    y₀ = 0.0km
    ø = 15inch
    ρ = 1.225kg/m^3
    g = 9.80665m/s^2

    # Calculated constants
    Aₚᵣ = π/4 * ø^2
    # Constants that we define as functions, because we may
    # want to modify them later in the same scripting session.
    α₀()  = 30°
    v₀()  = 1050m/s

    mₚ()   = 495kg
    C_s() = 0.4
    x´₀() = v₀() * cos(α₀())
    y´₀() = v₀() * sin(α₀())

    # Local tuple initial condition.
    u₀ = convert_to_mixed(x₀, y₀, x´₀(), y´₀())


    # Functions
    v(vx, vy) = √(vx^2 + vy^2)
    R(vx, vy) = 0.5∙ρ∙C_s()∙Aₚᵣ∙v(vx, vy)^2
    α(vx, vy) = atan(vy, vx)
    Rx(vx, vy) = R(vx, vy) * cos(α(vx, vy))
    Ry(vx, vy) = R(vx, vy) * sin(α(vx, vy))


    # Local tuple, i.e. the interesting degrees of freedom
    # and their derivatives
    function  Γ!(u´, u, p, t)
        x, y, x´, y´ = u
        # Calculate the acceleration for this step
        x´´ =     -Rx(x´, y´) / mₚ()
        y´´ = -1g -Ry(x´, y´) / mₚ()
        # Output
        u´ .= x´, y´, x´´, y´´
        u´
    end

    function solve_guarded(u₀)
        # Test the functions
        Γ!(u₀/s,u₀, nothing, nothing)
        prob = OrdinaryDiffEq.ODEProblem( Γ!,u₀,(0.0,60)s)
        solve(prob, OrdinaryDiffEq.Tsit5())
    end
    solve_guarded(u₀)
=#
end
