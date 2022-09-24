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

######################################
# Work note, for generating functions 
# similar to DEFAULT_SOLVE
# (might be helpful for GeneralMIRK4?)
######################################

"""
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

nothing