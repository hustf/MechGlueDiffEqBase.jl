# Test "multiplication" and "division" similar operations.
# For recursive ArrayPartition unitful mutable representations 
# of vectors, matrices and transposed versions of these.
# Informally, these are 'mixed' matrices and vectors.
using Test
import Logging
using Logging: LogLevel, with_logger, ConsoleLogger
using MechGlueDiffEqBase # exports ArrayPartition
using MechGlueDiffEqBase: is_reciprocal_symmetric, mul!
using MechanicalUnits: @import_expand, ∙, ustrip, unit
import MechanicalUnits: Unitfu
using Unitfu: DimensionError
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
###############################################
# Pre-multiply with the inverse of mixed matrix
###############################################
let 
    # The examples are based on http://www.georgehart.com/research/multanal.html
    w = [1 1; 1 10]cm 
    x = [1cm 1s; 1s 10cm] 
    y = [1cm 1cm∙s; 1cm/s 10cm]
    z = [1 1s; 1/s 10]
    @test_throws DimensionError w^2
    @test_throws DimensionError x^2
    @test y^2 == [2cm² 11cm²∙s; 11cm²∙s⁻¹ 101cm²]
    @test z^2== [2 11s;  11s⁻¹ 101 ]
    wm = convert_to_mixed(w) 
    xm = convert_to_mixed(x) 
    ym = convert_to_mixed(y)
    zm = convert_to_mixed(z)
    @test !is_reciprocal_symmetric(oneunit.(wm))
    @test !is_reciprocal_symmetric(oneunit.(xm))
    @test !is_reciprocal_symmetric(oneunit.(ym))
    @test is_reciprocal_symmetric(oneunit.(zm))
    @test_throws AssertionError convert_to_mixed([1.0 1.0s])
    b = convert_to_mixed([1.0, 1.0/s])
    #with_logger(Logging.ConsoleLogger(stderr, Logging.Debug;meta_formatter = locfmt)) do
        @test zm \ b == convert_to_mixed([1.0, 0.0/s])
    #end
    # Missing: test with transposed versions
end

#############################################
# Multiplication mixed matrix by mixed vector
#############################################
let
    A3 = [1 2 3; 4 5 6; 7 8 0]
    b3 = [1, 20, 300]
    c3 = [NaN, NaN, NaN]
    @test A3 * b3 == [1*1  + 2*20 + 3*300, 1904, 167]
    @test mul!(c3, A3, b3) == [941, 1904, 167] # Float64 on left, though, as NaN{Float64}
    c3 = [NaN, NaN, NaN]
    A3m = convert_to_mixed(A3)
    b3m = convert_to_mixed(b3)
    c3m = convert_to_mixed(c3)
    @test mul!(c3m, A3m, b3m) == [941, 1904, 167]
    @test A3m * b3m == [941, 1904, 167]
    A3u = [1kg 2kg∙cm 3s; 4s 5cm∙s 6s²∙kg⁻¹; 7cm⁻¹ 8 0s∙kg⁻¹∙cm⁻¹]
    b3u = [1cm, 20, 300kg∙cm∙s⁻¹]
    c3u = [1kg∙cm, 2s∙cm,3]
    A3mu = convert_to_mixed(A3u)
    b3mu = convert_to_mixed(b3u)
    c3mu = convert_to_mixed(c3u)
    @test mul!(c3mu, A3mu, b3mu) == [941kg∙cm, 1904cm∙s, 167]
    @test A3u * b3u == [941kg∙cm, 1904cm∙s, 167]
    @test A3mu * b3mu == [941kg∙cm, 1904cm∙s, 167]
end
#########################################################
# Multiplication, transposed mixed matrix by mixed vector
#########################################################
let
    A3 = [1 2 3; 4 5 6; 7 8 0]
    b3 = [1, 20, 300]
    c3 = [NaN, NaN, NaN]
    c3 = [NaN, NaN, NaN]
    A3m = convert_to_mixed(A3)
    b3m = convert_to_mixed(b3)
    c3m = convert_to_mixed(c3)
    A3u = [1kg 2kg∙cm 3s; 4s 5cm∙s 6s²∙kg⁻¹; 7cm⁻¹ 8 0s∙kg⁻¹∙cm⁻¹]
    b3u = [1cm, 20, 300kg∙cm∙s⁻¹]
    c3u = [1kg∙cm, 2s∙cm,3]
    A3mu = convert_to_mixed(A3u)
    b3mu = convert_to_mixed(b3u)
    c3mu = convert_to_mixed(c3u)
    @test_throws DimensionError transpose(A3u) * b3u
    @test_throws DimensionMismatch transpose(A3mu) * b3mu
    tA3mu = transpose(A3mu)
    d3 = convert_to_mixed([1cm, 20kg∙cm∙s⁻¹, 300kg∙cm²])
    e3 = convert_to_mixed([NaN∙kg∙cm, NaN∙kg∙cm², NaN∙s∙cm])
    @test mul!(e3, tA3mu, d3) == [2181.0kg∙cm, 2502.0kg∙cm², 123.0cm∙s]
    @test convert_to_array(tA3mu) * convert_to_array(d3) == [2181.0kg∙cm, 2502.0kg∙cm², 123.0cm∙s]
end
nothing