"""
    DIMENSIONAL_NLSOLVE(bvloss, u0)

    Function based on BoundaryValueDiffEq/src/algorithms.jl:11.

bvloss is intended to be a OnceDifferentiable function, a set of residuals
quantifying how well boundary conditions are fulfilled with starting conditions
u0. Zero residuals means fulfilled conditions.
"""
function DIMENSIONAL_NLSOLVE(bvloss, u0; resid0 = ArrayPartition([0.0], [0.0]))
    @debug "DIMENSIONAL_NLSOLVE" string(resid0) typeof(bvloss) fieldnames(typeof(bvloss))
    autodiff = :central
    inplace = !applicable(bvloss, u0)
    dloss = OnceDifferentiable(bvloss, u0, resid0; autodiff, inplace)
    xtol = zero(dloss.x_df)  # TODO consider dropping units, single parameter, keyword argument.
    ftol = 1.0e-8 .* oneunit.(dloss.F) # TODO consider dropping units, single parameter.
    @debug "DIMENSIONAL_NLSOLVE" (dloss isa OnceDifferentiable) string(xtol) string(ftol)
    iterations = 1000
    store_trace = false
    show_trace = false
    extended_trace = false
    factor = 1.0
    autoscale = true
    res = trust_region(dloss, u0, xtol, ftol, iterations,
        store_trace, show_trace, extended_trace, factor,
        autoscale)
    @debug "DIMENSIONAL_NLSOLVE" res
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
    bc = prob.bc
    u0 = deepcopy(prob.u0[1])
    resid0 = deepcopy(prob.u0[2])
    autodiff = :central
    inplace = true
    @debug "__solve extensions" string(u0) string(resid0)

    # Form a root finding function.
    bvloss = function (resid, minimizer)
        tmp_prob = remake(prob,u0=minimizer)
        sol = solve(tmp_prob, alg.ode_alg;kwargs...)
        bc(resid,sol,sol.prob.p,sol.t)
        nothing
    end
   # Call NLsolve/src/nlsolve/nlsolve.jl:1
    opt = alg.nlsolve(bvloss, u0; resid0)
    @debug "__solve opt" fieldnames(typeof(opt)) 
    sol_prob = remake(prob, u0=opt[1])
    @debug "__solve" sol_prob
    sol = solve(sol_prob, alg.ode_alg;kwargs...)
    if sol.retcode == opt[2]
        DiffEqBase.solution_new_retcode(sol,:Success)
    else
        DiffEqBase.solution_new_retcode(sol,:Failure)
    end
    sol
end
