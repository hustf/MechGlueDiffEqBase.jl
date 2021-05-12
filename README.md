# MechGlueDiffEqBase
Glue code for making [DiffEqBase](https://github.com/SciML/DiffEqBase.jl) work with units.

This defines

* how to calculate the vector norm 'ODE_DEFAULT_NORM' when the vector is given in units compatible with [Unitfu.jl](https://github.com/hustf/Unitfu.jl), from registy [M8](https://github.com/hustf/M8). The differential equation algorithms expects the norm to be unitless, as can be seen in e.g. step size estimators.

* type-stable and inferrable 'zero', 'value', 'UNITLESS_ABS2', 'similar'

These functions preserve types for mixed-unit vectors. E.g. [1.0kg, 2.0N, 3.0m/s, 4.0m/s].

This package also uses and reexports 'ArrayPartition' from [RecursiveArrayTools](https://github.com/SciML/RecursiveArrayTools.jl), which enables type-stable solution of equations with mixed units. It pre-compiles it with use of mixed unit vectors.

An example of inferrable, mixed units calculations with debugging is included in `test/test_5.jl`.

Note: The way error tolerances are defined is initially confusing, but good to know:

err_scaled = **error** / (**abstol** + norm(u) * **reltol**)

where **bold** indicates unitful objects.

Some functions are adaptions of corresponding code from [DiffEqBase](https://github.com/SciML/DiffEqBase.jl/blob/6bb8830711e729ef513f2b1beb95853e4a691375/src/init.jl).
