# These extend functions in NLSolversbase
# NLSolversBase\src\objective_types\abstract.jl:19
alloc_DF(x::MixedCandidate, F::MixedCandidate) = _alloc_DF(mixed_array_trait(x), mixed_array_trait(F), x, F) 

function _alloc_DF(::VecMut, ::VecMut, x, F)
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

################
# More dead code
################
"""
    unit_of_mixed(A::MixedContent(N)) where N
Extract FreeUnits in a similar struture to A. Unfortunately, the outermost
ArrayPartition type is 'Any'.
"""
function unit_of_mixed(A::MixedContent(N)) where N
    throw("unused1")
    ro(i) = ArrayPartition(unit.(A[i, :]))
    ArrayPartition((ro(i) for i = 1:N)...)
end

"""
    convert_to_ArrayPartition(A::AbstractArray{<:FreeUnits})
    -> nested immutable ArrayPartition

`A` is typically Matrix{<:FreeUnits} or Vector{<:FreeUnits}.
"""
function convert_to_arraypartition(A::AbstractArray{<:FreeUnits})
    throw("Not used, perhaps if ndims is extended it will be useful")
    #if is_square_matrix_mutable(A) || is_vector_mutable_stable(A)
    #    A
    if ndims(A) == 2
        @assert size(A)[1] > 1 "Cannot convert from row vectors"
        ArrayPartition{FreeUnits}(map(eachrow(A)) do rw
            ArrayPartition{FreeUnits}(map(rw) do el
                el
            end...)
        end...) #::MixedCandidate
    elseif ndims(A) == 1
        ArrayPartition(map(x-> x, A)...)
    else
        throw(DimensionMismatch())
    end
end

###########
# Dead code
###########
"This could extend Base.(*), but we would want to examine the Julia conventions better first."
function matmulvec(A::AbstractMatrix{T}, x::AbstractVector{T}) where T<:FreeUnits
    throw("Not used")
    n = size(A)[1]
    @assert n == size(A)[2]
    @assert n == size(x)[1]
    Aap = convert_to_arraypartition(A)
    @debug "matmulvec" Aap

    #for i in 1:n
    #    for j in 1:n
    #        @debug "matmulvec" i j A[i, j]  x[j] A[i, j] * x[j]
    #    end
    #end
    #r
    matmulvec(Aap, x)


end
function matmulvec(A::ArrayPartition, x::ArrayPartition)
    throw("unused1")
    n = size(x)[1]
    @debug "matmulvec" typeof(A) typeof(x) n
    function pr(i, j)
        Ael = A[i, j]
        xel = x[j]
        Ael * xel
    end
    function ro(i)
        pr1 = pr(i, 1)
        for j = 2:n
            if pr(i, j) != pr1
                throw(DimensionError("row = $i col 1 $(pr(i,j))", "row = $i col $j $(pr(i,j))" ))
            end
        end
        pr1
    end
     ma = ArrayPartition((ro(i) for i = 1:n)...)
     @debug "matmulvec" ma
     ma
end