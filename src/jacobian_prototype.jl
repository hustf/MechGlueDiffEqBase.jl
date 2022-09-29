####################
# Jacobian prototype
####################

"""
    jacobian_prototype_zero(x, vecfx)
-> type-stable, mutable Jacobian container prototype.
Assumes x is a vector of arguments to function f, and vecfx = f(x).
The Jacobian associates elements of x with columns, and
elements of vecfx with rows. If both vectors have elements with units,
each element in the Jacobian matrix could have unique units. This container
is intended to help the compiler making efficient machine code by foretelling
units, and can be used as if it was an ordinary Matrix{Any}.
"""
jacobian_prototype_zero(x, vecfx) = zero(jacobian_prototype_nan(x, vecfx))

jacobian_prototype_nan(x::MixedCandidate, vecfx::MixedCandidate) = 
    jacobian_prototype_nan(mixed_array_trait(x), mixed_array_trait(vecfx), x, vecfx)

function jacobian_prototype_nan(::VecMut, ::VecMut, x, vecfx)
    @debug "jacobian_prototypes/jacobian_prototype_nan:21" string(x) string(vecfx) maxlog=2 
    # Mutable input
    typednan = (xel, fxel) -> NaN * (fxel / xel)
    genrow(fxel) = map(xel -> typednan(xel, fxel) , x)
    Jint = map(genrow, vecfx)
    ArrayPartition(Jint...)
end
