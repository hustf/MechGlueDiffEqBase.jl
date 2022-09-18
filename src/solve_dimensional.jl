# Type based on NLsolve/src/solver_state_results.jl 
mutable struct SolverResultsDimensional{N, T<:Real}
    method::String
    initial_x::RW(N)
    zero::RW(N)
    residual_norm::T
    iterations::Int
    x_converged::Bool
    xtol::T
    f_converged::Bool
    ftol::T
    trace::NLsolve.SolverTrace
    f_calls::Int
    g_calls::Int
end

function converged(r::SolverResultsDimensional)
    return r.x_converged || r.f_converged
end

function Base.show(io::IO, r::SolverResultsDimensional)
    @printf io "Results of dimensional Nonlinear Solver Algorithm\n"
    @printf io " * Algorithm: %s\n" r.method
    @printf io " * Starting Point: %s\n" string(r.initial_x)
    @printf io " * Zero: %s\n" string(r.zero)
    @printf io " * Inf-norm of residuals: %f\n" r.residual_norm
    @printf io " * Iterations: %d\n" r.iterations
    @printf io " * Convergence: %s\n" converged(r)
    @printf io "   * |x - x'| < %.1e: %s\n" r.xtol r.x_converged
    @printf io "   * |f(x)| < %.1e: %s\n" r.ftol r.f_converged
    @printf io " * Function Calls (f): %d\n" r.f_calls
    @printf io " * Jacobian Calls (df/dx): %d" r.g_calls
    return
end



















###########
# Dead code
###########

"""
    DIMENSIONAL_NLSOLVE(bvloss, u0)

    Function based on BoundaryValueDiffEq/src/algorithms.jl:11.

The solution process repeatedly calls `bc!(u, sol, p, t)`, where `u` starts out as a copy of `u0` and ends up as `u == zero(u)`

# Arguments
- `u0` initial condition to `f´!(du, u, p, t)`, defined in `BVProblem(f´!, bc!, u0, tspan, p)`.
- `bvloss` is a root finding function for NLSolve, defined by the caller, `DiffEqBase.__solve`.

We internally extract the captured variables of `bvloss` as fields:
- `bvloss.kwargs`: 'dtmax' is internally stripped of units. `solve(..;dtmax= 0.05s`) -> `:dtmax => 0.05`
- `bvloss.prob`::BVProblem with fields:
    - `.f`::ODEFunction, see inline docs.
        -> `bvloss.prob.f.f`:: `f´!`
    - `.bc`: `bc!`
    - `.tspan`
    - `.p`: The parameters for the problem. Defaults to `NullParameters`.
    - `kwargs`
- `bvloss.alg` - e.g. type `Shooting` or `GeneralMIRK4`
    - `.ode_alg`
    - `.nlsolve`
- `bvloss.sol`::Core.Box(#undef)... a placeholder, at first call.
- `bvloss.bc`
"""
function DIMENSIONAL_NLSOLVE(bvloss, u0)
    throw("dead code, does the same as DEFAULT_SOLVE. May be useful later?")
    # This is identical to DEFAULT_NLSOLVE now. TODO: Delete. Use inline doc elsewhere?


    #@debug "DIMENSIONAL_NLSOLVE:26" bvloss.kwargs bvloss.prob bvloss.alg bvloss.sol bvloss.bc maxlog = 2
    #@debug "DIMENSIONAL_NLSOLVE:27" bvloss.prob bvloss.prob.f bvloss.prob.bc bvloss.prob.tspan bvloss.prob.p bvloss.prob.kwargs maxlog = 2
    #@debug "DIMENSIONAL_NLSOLVE:28" fieldnames(typeof(bvloss.prob.f)) maxlog = 2
    #@debug "DIMENSIONAL_NLSOLVE:29" typeof(bvloss.prob.f) maxlog = 2
    #autodiff = :central
    #inplace = !applicable(bvloss, string(u0))
    #@debug "DIMENSIONAL_NLSOLVE:34" bvloss.sol bvloss.prob.p bvloss.prob.tspan fieldnames(typeof(bvloss.alg)) maxlog = 2
    #inisol =  solve(bvloss.prob, bvloss.alg.ode_alg; bvloss.kwargs...);
    #F = bvloss.prob.bc(u0, inisol, bvloss.prob.p, bvloss.prob.tspan[1])
    @debug "DIMENSIONAL_NLSOLVE:37" string(u0) maxlog = 2
    res = NLsolve.nlsolve(bvloss, u0)

    #=
    dloss = OnceDifferentiable(bvloss, u0, resid0; autodiff, inplace)
    xtol = zero(dloss.x_df)  # TODO consider dropping units, single parameter, keyword argument.
    ftol = 1.0e-8 .* oneunit.(dloss.F) # TODO consider dropping units, single parameter.

    iterations = 1000
    store_trace = false
    show_trace = false
    extended_trace = false
    factor = 1.0
    autoscale = true
    res = trust_region(dloss, u0, xtol, ftol, iterations,
        store_trace, show_trace, extended_trace, factor,
        autoscale)
    @debug "DIMENSIONAL_NLSOLVE res"
    =#
    (res.zero, res.f_converged)
end
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
    throw("That triggers me.")
    bc = prob.bc
    u0 = deepcopy(prob.u0[1])
    resid0 = deepcopy(prob.u0[2])
    autodiff = :central
    inplace = true
    @debug "__solve extensions" string(u0) string(resid0)

    # Form a root finding function.
    bvloss = function (resid, minimizer)
        tmp_prob = remake(prob,u0 = minimizer)
        sol = solve(tmp_prob, alg.ode_alg;kwargs...)
        bc(resid,sol,sol.prob.p,sol.t)
        nothing
    end
   # Call NLsolve/src/nlsolve/nlsolve.jl:1
    opt = alg.nlsolve(bvloss, u0; resid0)
    @debug "__solve opt" fieldnames(typeof(opt))
    sol_prob = remake(prob, u0 = opt[1])
    @debug "__solve" sol_prob
    sol = solve(sol_prob, alg.ode_alg;kwargs...)
    if sol.retcode == opt[2]
        DiffEqBase.solution_new_retcode(sol,:Success)
    else
        DiffEqBase.solution_new_retcode(sol,:Failure)
    end
    sol
end
