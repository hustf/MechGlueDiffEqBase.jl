# These extend functions in NLSolversbase
# NLSolversBase\src\objective_types\abstract.jl:19
alloc_DF(x::MixedCandidate, F::MixedCandidate) = alloc_DF(mixed_array_trait(x), mixed_array_trait(F), x, F) 

function alloc_DF(::VecMut, ::VecMut, x, F)
    @debug "_alloc_DF:6" string(x) string(F) maxlog = 2
    jacobian_prototype_nan(x, F)
end


"Extend NLSolversbase/oncedifferentiable.jl:227, called from oncedifferentiable.jl:97"
function OnceDifferentiable(f, df, fdf,
    x::RW(N),
    F::RW(N),
    DF::AbstractArray = alloc_DF(x, F);
    inplace = true) where N
    @debug "OnceDifferentiable:25" string(x) string(F) string(DF) maxlog = 2
    f = f!_from_f(f, F, inplace)
    df! = df!_from_df(df, F, inplace)
    fdf! = fdf!_from_fdf(fdf, F, inplace)
    x_f = copy(NaN * x)
    x_df = copy(x_f)
    OnceDifferentiable(f, df!, fdf!, copy(F), copy(DF), x_f, x_df, [0,], [0,])
end