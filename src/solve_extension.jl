# Extends __solve, defined in BoundaryValueDiffEq/src/solve.jl:4
# That versions expects prob.u0 to contain the argument prototype for the ODEFunction, typically u0,
# and it copies that argument to also be used for the intital residual value of the loss function.
# 
# This extension is dispatched to when prob.u0 contains a tuple. 
# The first element in the tuple is interpreted as normal, i.e. u0.
# The second element is interpreted as the initial residual value.
# It is nice to have both, because they often have different units,
# and this way we can determine the Jacobian of the loss function.
function __solve(prob::BVProblem{U}, alg::Shooting; kwargs...) where {U<:Tuple{<:ArrayPartition, <:ArrayPartition}}
    bc = prob.bc
    u0 = deepcopy(prob.u0[1])
    resid0 = deepcopy(prob.u0[2])
    @debug "__solve extensions" string(u0) string(resid0)

    # Form a root finding function.
    loss = function (resid, minimizer)
        uEltype = eltype(minimizer)
        tmp_prob = remake(prob,u0=minimizer)
        sol = solve(tmp_prob,alg.ode_alg;kwargs...)
        bc(resid,sol,sol.prob.p,sol.t)
        nothing
    end
    opt = alg.nlsolve(loss, u0)
    sol_prob = remake(prob, u0=opt[1])
    sol = solve(sol_prob, alg.ode_alg;kwargs...)
    if sol.retcode == opt[2]
        DiffEqBase.solution_new_retcode(sol,:Success)
    else
        DiffEqBase.solution_new_retcode(sol,:Failure)
    end
    sol
end
function nlsolve(f,
    initial_x::RW(N);
    method::Symbol = :trust_region,
    autodiff = :central,
    inplace = !applicable(f, initial_x),
    kwargs...) where N
    if method in (:anderson, :broyden)
        df = NonDifferentiable(f, initial_x, copy(initial_x); inplace=inplace)
    else
        @debug "nlsolve extended" initial_x autodiff inplace kwargs methods(f)
        df = OnceDifferentiable(f, initial_x, copy(initial_x); autodiff=autodiff, inplace=inplace)
    end

    nlsolve(df, initial_x; method = method, kwargs...) 
end
#=
BVProblem{ArrayPartition{Quantity{Float64}, Tuple{Vector{Float64}, Vector{Quantity{Float64,  ᵀ⁻¹, Unitfu.FreeUnits{(s⁻¹,),  ᵀ⁻¹, nothing}}}}},
           Tuple{Quantity{Float64,  ᵀ, Unitfu.FreeUnits{(s,),  ᵀ, nothing}}, Quantity{Float64,  ᵀ, Unitfu.FreeUnits{(s,),  ᵀ, nothing}}}, 
           true, 
           Tuple{Quantity{Int64,  ᴸ∙ ᵀ⁻², Unitfu.FreeUnits{(cm, s⁻²),  ᴸ∙ ᵀ⁻², nothing}}, Quantity{Float64,  ᴸ, Unitfu.FreeUnits{(cm,),  ᴸ, nothing}}}, 
           ODEFunction{true, typeof(simplependulum´!), LinearAlgebra.UniformScaling{Bool}, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, Nothing, typeof(SciMLBase.DEFAULT_OBSERVED), Nothing}, 
           typeof(bc2!), 
           SciMLBase.StandardBVProblem, 
           Base.Pairs{Symbol, Union{}, Tuple{}, NamedTuple{()"#undef", Tuple{}}}
           }
=#