# Test adaptions to NLSolversBase, NLSolve
using Test
import Logging
using Logging: LogLevel, with_logger, ConsoleLogger
using MechGlueDiffEqBase
using MechGlueDiffEqBase: nlsolve, converged
using MechanicalUnits: @import_expand, ∙
import NLsolve

@import_expand(cm, kg, s)
"""
Debug formatter, highlight NLSolversBase. To use:
```
with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do 
    @test ...
end
```
"""
function locfmt(level::LogLevel, _module, group, id, file, line)
    @nospecialize
    if repr(_module) == "FiniteDiff"
        color = :green
    elseif repr(_module) == "Main"
        color = :176
    elseif repr(_module) ==  "MechGlueDiffEqBase"
        color = :magenta
    else
        color = :blue
    end
    prefix = string(level == Logging.Warn ? "Warning" : string(level), ':')
    suffix::String = ""
    Logging.Info <= level < Logging.Warn && return color, prefix, suffix
    _module !== nothing && (suffix *= "$(_module)")
    if file !== nothing
        _module !== nothing && (suffix *= " ")
        suffix *= Base.contractuser(file)::String
        if line !== nothing
            suffix *= ":$(isa(line, UnitRange) ? "$(first(line))-$(last(line))" : line)"
        end
    end
    !isempty(suffix) && (suffix = "@ " * suffix)
    return color, prefix, suffix
end
######################
# 1 Utilities, NLsolve
######################
@test_throws MechGlueDiffEqBase.NLsolve.IsFiniteException check_isfinite([1,NaN])
@test_throws MechGlueDiffEqBase.NLsolve.IsFiniteException check_isfinite([1,Inf])
@test_throws MechGlueDiffEqBase.NLsolve.IsFiniteException check_isfinite(
    convert_to_mixed([1,Inf]))
@test_throws MechGlueDiffEqBase.NLsolve.IsFiniteException check_isfinite(
    convert_to_mixed([1s,Inf]))
@test_throws MechGlueDiffEqBase.NLsolve.IsFiniteException check_isfinite([1 2;Inf 4])
@test_throws MechGlueDiffEqBase.NLsolve.IsFiniteException check_isfinite(
    convert_to_mixed([1 2;Inf 4]))
@test_throws MechGlueDiffEqBase.NLsolve.IsFiniteException check_isfinite(
    convert_to_mixed([1s 2;Inf 4]))
#############################
# 2 Newton trust region solve
#   Vectors
#############################
function f_2by2!(F, x)
    F[1] = (x[1]+3)*(x[2]^3-7)+18
    F[2] = sin(x[2]*exp(x[1])-1)
end
F1 = [10.0, 20.0]
# Evaluate implicitly at known zero
f_2by2!(F1, [0,1]) 
@test F1 == [0.0, 0.0]
# OnceDifferentiable contains both f and df. We give prototype arguments.
xprot1 = [NaN, NaN]
df1 = OnceDifferentiable(f_2by2!, xprot1, F1; autodiff = :central)

@test NLsolve.NewtonTrustRegionCache(df1) isa NLsolve.AbstractSolverCache
@test MechGlueDiffEqBase.LenNTRCache(df1) isa NLsolve.AbstractSolverCache
# Start at a point outside zero, iterate arguments until function value is zero.
r = nlsolve(df1, [ -0.5, 1.4], method = :trust_region, autoscale = true)
@test converged(r)
# Did we find the correct arguments?
@test r.zero ≈ [ 0, 1]
@test r.iterations == 4
#############################
# 2 Newton trust region solve
#  ArrayPartition
#############################
F2 = convert_to_mixed([10.0, 20.0])
# Evaluate implicitly at known zero
f_2by2!(F2, convert_to_mixed([0,1])) 
@test F2 == convert_to_mixed([0.0, 0.0])
# df includes both f_2by2, and its 'derivative'. We supply argument prototypes to both.
xprot2 = convert_to_mixed([NaN, NaN])
df2 = OnceDifferentiable(f_2by2!, xprot2, F2; autodiff = :central)
@test NLsolve.NewtonTrustRegionCache(df2) isa NLsolve.AbstractSolverCache
@test MechGlueDiffEqBase.LenNTRCache(df2) isa NLsolve.AbstractSolverCache
# Start at a point outside zero, iterate arguments until function value is zero.

with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do
    nlsolve(df2, convert_to_mixed([ -0.5; 1.4]), method = :trust_region, autoscale = true)
end

r = nlsolve(df2, convert_to_mixed([ -0.5, 1.4]), method = :trust_region, autoscale = true)
@test converged(r)
# Did we find the correct arguments?
@test r.zero ≈ [ 0, 1]
@test r.iterations == 4

#############################
# 3 Newton trust region cache
#  ArrayPartition dimensional
#############################
function f_2by2a!(F, x)
    F[1] = (x[1]+3kg)*(x[2]^3-7s^3)+18kg∙s³      
    F[2] = sin(x[2]*exp(x[1]/kg)/s-1)s
end
F3 = convert_to_mixed([10.0kg∙s³, 20.0s])
# Evaluate implicitly at known zero
f_2by2a!(F3, convert_to_mixed([0kg,1s])) 
@test F3 == convert_to_mixed([0.0kg∙s³, 0.0s])
# df includes both f_2by2, and its 'derivative'. We supply argument prototypes to both.
xprot3 = convert_to_mixed([NaN∙kg, NaN∙s])
df3 = OnceDifferentiable(f_2by2a!, xprot3, F3; autodiff = :central)
@test_throws MethodError NLsolve.NewtonTrustRegionCache(df3)
@test MechGlueDiffEqBase.LenNTRCache(df3) isa NLsolve.AbstractSolverCache
# Start at a point outside zero, iterate arguments until function value is zero.
#=
#with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do 
#nlsolve(df3, convert_to_mixed([ -0.5∙kg; 1.4∙s]), method = :trust_region, autoscale = true)
#end

r = nlsolve(df3, convert_to_mixed([ -0.5∙kg; 1.4∙s]), method = :trust_region, autoscale = true)
@test converged(r)
# Did we find the correct arguments?
@test r.zero ≈ [ 0, 1]
@test r.iterations == 4
=#
nothing