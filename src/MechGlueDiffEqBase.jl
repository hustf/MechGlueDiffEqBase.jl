module MechGlueDiffEqBase
import Unitfu: AbstractQuantity, Quantity, ustrip
import DiffEqBase: value, ODE_DEFAULT_NORM, UNITLESS_ABS2
export value, ODE_DEFAULT_NORM, UNITLESS_ABS2, Unitfu, AbstractQuantity, Quantity


import RecipesBase
import RecipesBase.@recipe
import SciMLBase
import SciMLBase: AbstractTimeseriesSolution, RecipesBase
import SciMLBase: AbstractDiscreteProblem, AbstractRODESolution, SensitivityInterpolation
import SciMLBase: getsyms, interpret_vars, cleansyms
import SciMLBase: diffeq_to_arrays, issymbollike, getindepsym_defaultt
import SciMLBase: DEFAULT_PLOT_FUNC


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

# This recipe is copied from SciMLBase, and only the signature is adopted
# for use with unitful solutions. It can perhaps be deleted now, and the link to RecipesBase be deleted.
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
    syms = getsyms(sol)
    int_vars = interpret_vars(vars,sol,syms)
    strs = cleansyms(syms)
  
    tscale = get(plotattributes, :xscale, :identity)
    plot_vecs,labels = diffeq_to_arrays(sol,plot_analytic,denseplot,
                                        plotdensity,tspan,axis_safety,
                                        vars,int_vars,tscale,strs)
  
    tdir = sign(sol.t[end]-sol.t[1])
    xflip --> tdir < 0
    seriestype --> :path
  
    # Special case labels when vars = (:x,:y,:z) or (:x) or [:x,:y] ...
    if typeof(vars) <: Tuple && (issymbollike(vars[1]) && issymbollike(vars[2]))
      xguide --> issymbollike(int_vars[1][2]) ? Symbol(int_vars[1][2]) : strs[int_vars[1][2]]
      yguide --> issymbollike(int_vars[1][3]) ? Symbol(int_vars[1][3]) : strs[int_vars[1][3]]
      if length(vars) > 2
        zguide --> issymbollike(int_vars[1][4]) ? Symbol(int_vars[1][4]) : strs[int_vars[1][4]]
      end
    end
  
    if (!any(issymbollike,getindex.(int_vars,1)) && getindex.(int_vars,1) == zeros(length(int_vars))) ||
       (!any(issymbollike,getindex.(int_vars,2)) && getindex.(int_vars,2) == zeros(length(int_vars))) ||
       all(t->Symbol(t)==getindepsym_defaultt(sol),getindex.(int_vars,1)) || all(t->Symbol(t)==getindepsym_defaultt(sol),getindex.(int_vars,2))
      xguide --> "$(getindepsym_defaultt(sol))"
    end
    if length(int_vars[1]) >= 3 && ((!any(issymbollike,getindex.(int_vars,3)) && getindex.(int_vars,3) == zeros(length(int_vars))) ||
       all(t->Symbol(t)==getindepsym_defaultt(sol),getindex.(int_vars,3)))
      yguide --> "$(getindepsym_defaultt(sol))"
    end
    if length(int_vars[1]) >= 4 && ((!any(issymbollike,getindex.(int_vars,4)) && getindex.(int_vars,4) == zeros(length(int_vars))) ||
       all(t->Symbol(t)==getindepsym_defaultt(sol),getindex.(int_vars,4)))
      zguide --> "$(getindepsym_defaultt(sol))"
    end
  
    if (!any(issymbollike,getindex.(int_vars,2)) && getindex.(int_vars,2) == zeros(length(int_vars))) ||
        all(t->Symbol(t)==getindepsym_defaultt(sol),getindex.(int_vars,2))
      if tspan === nothing
        if tdir > 0
          xlims --> (sol.t[1],sol.t[end])
        else
          xlims --> (sol.t[end],sol.t[1])
        end
      else
        xlims --> (tspan[1],tspan[end])
      end
    else
      mins = minimum(sol[int_vars[1][2],:])
      maxs = maximum(sol[int_vars[1][2],:])
      for iv in int_vars
        mins = min(mins,minimum(sol[iv[2],:]))
        maxs = max(maxs,maximum(sol[iv[2],:]))
      end
      xlims --> ((1-sign(mins)*axis_safety)*mins,(1+sign(maxs)*axis_safety)*maxs)
    end
  
    # Analytical solutions do not save enough information to have a good idea
    # of the axis ahead of time
    # Only set axis for animations
    if sol.tslocation != 0 && !(typeof(sol) <: AbstractAnalyticalSolution)
      if all(getindex.(int_vars,1) .== DEFAULT_PLOT_FUNC)
        mins = minimum(sol[int_vars[1][3],:])
        maxs = maximum(sol[int_vars[1][3],:])
        for iv in int_vars
          mins = min(mins,minimum(sol[iv[3],:]))
          maxs = max(maxs,maximum(sol[iv[3],:]))
        end
        ylims --> ((1-sign(mins)*axis_safety)*mins,(1+sign(maxs)*axis_safety)*maxs)
  
        if length(int_vars[1]) >= 4
          mins = minimum(sol[int_vars[1][4],:])
          maxs = maximum(sol[int_vars[1][4],:])
          for iv in int_vars
            mins = min(mins,minimum(sol[iv[4],:]))
            maxs = max(mins,maximum(sol[iv[4],:]))
          end
          zlims --> ((1-sign(mins)*axis_safety)*mins,(1+sign(maxs)*axis_safety)*maxs)
        end
      end
    end
  
    label --> reshape(labels,1,length(labels))
    println("----------------------------------MechGlueDiffEqBase")
    (plot_vecs...,)
end

end
