using MechanicalUnits
using MechGlueDiffEqBase
using MechGlueDiffEqBase: determinant_dimension, determinant,  @printf


# Example from test_012.jl
E = 200GPa
I = 2e6mm⁴
A = 1000mm²
l = 100cm
# `NoUnits`` simplifies "fraction units" like cm/mm.
kx = E∙A / l |> NoUnits |> kN
ky = 12∙E∙I / l^3 |> NoUnits |> kN
kθ = 2∙E∙I / l |> NoUnits |> kN
kθy = 6∙E∙I / l^2  |> NoUnits |> kN

# Elementary beam stiffness matrix, six degrees of freedom
K = convert_to_mixed([kx        0kN/mm     0kN        -kx       0kN/mm     0kN;
                    0kN/mm     ky        -kθy      0kN/mm     -ky       -kθy;
                    0kN        -kθy      2∙kθ      0kN        kθy       kθ  ;
                    -kx       0kN/mm     0kN        kx        0kN/mm     0kN  ;
                    0kN/mm     -ky       kθy       0kN/mm     ky        kθy ;
                    0kN        -kθy      kθ        0kN        kθy       2∙kθ])

# 1) The system is dimensionally sound. We would get an error otherwise
# 2) MechanicalUnits defines unicode exponents up to 4
@assert determinant_dimension(K) == 𝐋⁴∙𝐌^6 ∙𝐓^-12
# 3) The matrix is not invertible. 
@assert determinant(K) == 0.0kN^6∙mm⁻²
# The next step here would be to drop the known rows and columns, and the deteminant is >0.
# If you don't know anything, it's hopeless.

# We can rearrange rows (similarly columns) by temporarily converting to Any:
Ka = convert_to_array(K)
Base.swaprows!(Ka, 1, 2)
Ks = convert_to_mixed(Ka)
@assert Ks[1, :] == K[2, :]
# We could do linear solving by Gauss elimination.
determinant(Ks) == determinant(K)


# We can't generally add rows to each other:
K[1, :] .+ K[2, :]
K[1, :] .+ K[3, :]
# That means we can't do LU factorization (except by cheating).
# The way to cheat is probably to 
# 1) Check determinant_dimension or determinant value > 0
# 2) Drop units
# 3) Do linear solve with any nice method
# 4) Add units back in
