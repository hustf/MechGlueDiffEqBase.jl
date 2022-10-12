module MechGlueDiffEqBase
# TODO: Don't import. Use!
import Base: similar, inv, +, -, *, \
import Base: show, summary, print, size, ndims
import Base: axes, copy, getindex, setindex!
using Base: array_summary, throw_boundserror, setindex
import Base.Iterators
import Base.Iterators: zip
import Base.Broadcast
import Base.Broadcast: BroadcastStyle, combine_styles
using Base.Broadcast: broadcasted, IndexStyle
import Printf
using Printf: @printf

import LinearAlgebra
import LinearAlgebra: mul!
using LinearAlgebra: require_one_based_indexing
using LinearAlgebra: AdjOrTransAbsVec, Transpose

import MechanicalUnits
import MechanicalUnits: numtype
using MechanicalUnits: Quantity, ustrip, norm, unit, zero, ∙, uconvert
using MechanicalUnits: uconvert, dimension, ∙
using MechanicalUnits: Dimensions, Dimension, FreeUnits, NoUnits, DimensionlessQuantity
using MechanicalUnits.Unitfu: AbstractQuantity # Consider dropping this (used below)

import DiffEqBase
import DiffEqBase: value, ODE_DEFAULT_NORM, UNITLESS_ABS2
using  DiffEqBase: @muladd, __solve, BVProblem, solve 

import BoundaryValueDiffEq
using BoundaryValueDiffEq: Shooting

import ArrayInterfaceCore
import RecursiveArrayTools
import RecursiveArrayTools: unpack
using RecursiveArrayTools: ArrayPartition, ArrayPartitionStyle, npartitions, unpack_args

import FiniteDiff
import FiniteDiff: compute_epsilon, finite_difference_derivative
import FiniteDiff: finite_difference_jacobian, finite_difference_jacobian!, JacobianCache
using FiniteDiff: default_relstep, fdtype_error, _vec

import NLSolversBase
import NLSolversBase: alloc_DF, OnceDifferentiable
using NLSolversBase: f!_from_f, df!_from_df, fdf!_from_fdf, value_jacobian!!

import NLsolve
import NLsolve: trust_region, check_isfinite, converged
using NLsolve: NewtonTrustRegionCache, dogleg!, nlsolve, assess_convergence

# TODO only export extended and defined functions and types?
export value, ODE_DEFAULT_NORM, UNITLESS_ABS2, Quantity, FreeUnits
export norm , ArrayPartition, similar, zero, compute_epsilon
export jacobian_prototype_zero, jacobian_prototype_nan
export finite_difference_derivative, finite_difference_jacobian, show, summary, print
export MixedCandidate, convert_to_array, JacobianCache
export numtype, alloc_DF, mixed_array_trait, convert_to_mixed
export MatSqMut, VecMut, NotMixed, NotMixed, Single, Empty
export is_square_matrix_mutable, is_vector_mutable_stable
export OnceDifferentiable, check_isfinite


# The file "io_traits_conversion.jl" defines, by traits, 
# certain variants of RecursiveArrayTools.ArrayPartition.
# We dispatch on these traits.
#
# Defines for external use:
#   convert_to_mixed
#   convert_to_array
#   is_square_matrix_mutable
#   is_vector_mutable_stable
# Defines for internal dispatch:
#   mixed_array_trait
# Defines type constants for internal use: 
#   Q, E, RW(N), MatrixMixedCandidate, MixedCandidate,UnionVeqSqMut
# Defines (trait) types: 
#   Mixed, MatSqMut, VecMut, NotMixed, Single, Empty
# Defines for internal use:
#   vpack
#   ArrayPartition_from_single_element_vectors
#   print_as_mixed
#   getindex_of_transposed_mixed (TODO inline it)
#   setindex!_of_transposed_mixed
#   IndexStyle_of_transposed_mixed
# Extends:
#   Base.size
#   Base.ndims
#   Base.axes
#   Base.getindex 
#   Base.setindex!
#   Base.print
#   Base.show
#   Base.summary
#   Base.Broadcast._IndexStyle (??)
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

# MechGlueDiffEqBase.jl extends:
#   UNITLESS_ABS2
#   value
#   ODE_DEFAULT_NORM
#   Base.similar

# This extends:
#   Base.Broadcast.BroadcastStyle
#   Base.Broadcast.combine_styles
#   Base.copy
#   Base.map
#   Base.Iterators.zip
#   RecursiveArrayTools.unpack
# And defines for internal use
#   maprow
#   _combine_styles
#   _map
#   _zip
#   _getindex

include("broadcast_mixed_matrix.jl")

# This extends:
#   FiniteDiff.compute_epsilon
#   FiniteDiff.finite_difference_derivative
include("derivatives_dimensional.jl")

# This defines:
#   jacobian_prototype_zero
#   jacobian_prototype_nan
include("jacobian_prototype.jl")

# This extends:
#   FiniteDiff.JacobianCache
include("jacobian_cache.jl")

# This extends
#   NLSolversBase.alloc_DF
#   NLSolversBase.OnceDifferentiable
include("once_differentiable.jl")

# This extends
#   LinearAlgebra.mul!
#   Base.(*)
#   Base.(+)
#   Base.(-)
#   Base.(\)
#   Base.inv
# And defines
#   premul_inv (used by (\))
include("multiply_divide.jl")

# This defines a similar structure to NLSolve/SolverResults:
#   SolverResultsDimensional
# It extends:
#   NLSolve.converged
#   Base.show
include("solve_dimensional.jl")

# This defines a new type <: NLsolve.AbstractSolverCache:
#   LenNTRCache
#   dogleg_dimensional!
# It extends:
#   NLSolve.trust_region
include("trustregion_dimensional.jl")

# This extends:
#   FiniteDiff.finite_difference_jacobian
#   FiniteDiff.finite_difference_jacobian!
include("jacobians_dimensional.jl")

# This extends:
#   Unitfu.numtype
#   NLSolve.check_isfinite
# And defines:
#   determinant
#   determinant_dimension
include("utils_dimensional.jl")


end
