module MechGlueDiffEqBase
import Unitfu: AbstractQuantity, Quantity
import DiffEqBase: value, ODE_DEFAULT_NORM, UNITLESS_ABS2
import SciMLBase
import SciMLBase: AbstractTimeseriesSolution, RecipesBase
import SciMLBase: AbstractDiscreteProblem, AbstractRODESolution, SensitivityInterpolation
import RecipesBase.@recipe
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


@recipe function f(sol::AbstractTimeseriesSolution{T, N, A};
    plot_analytic=false,
    denseplot = (sol.dense ||
                 typeof(sol.prob) <: AbstractDiscreteProblem) &&
                 !(typeof(sol) <: AbstractRODESolution) &&
                 !(hasfield(typeof(sol),:interp) &&
                   typeof(sol.interp) <: SensitivityInterpolation),
    plotdensity = min(Int(1e5),sol.tslocation==0 ?
                 (typeof(sol.prob) <: AbstractDiscreteProblem ?
                 max(1000,100*length(sol)) :
                 max(1000,10*length(sol))) :
                 1000*sol.tslocation),
    tspan = nothing, axis_safety = 0.1,
    vars=nothing) where {T<:Quantity, N, A<:Array{<:Quantity}}

    print("----------------------------------MechGlueDiffEqBase")
    return Unifu.ustrip(sol.u), Unifu.ustrip(sol.t)
end
        # T = Quantity{Float64,  ᴸ∙ ᴹ∙ ᵀ⁻², Unitfu.FreeUnits{(N,),  ᴸ∙ ᴹ∙ ᵀ⁻², nothing}}
    # N = 1
    # A = Vector{Quantity{Float64,  ᴸ∙ ᴹ∙ ᵀ⁻², Unitfu.FreeUnits{(N,),  ᴸ∙ ᴹ∙ ᵀ⁻², nothing}}}


end
