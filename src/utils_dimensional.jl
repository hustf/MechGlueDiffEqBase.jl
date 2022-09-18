# Extend Unitifu function for non-quantities
numtype(x::T) where {T<:Union{DataType, Number}} = isbitstype(T) ? T : x
# Extends NLsolve.jl/src/ultils.jl:21
check_isfinite(x::MixedCandidate) = check_isfinite(mixed_array_trait(x), x)

function check_isfinite(::VecMut, x)
    if any(any.(!isfinite, x))
        i = findall(!isfinite, x)
        throw(NLsolve.IsFiniteException(i))
    end
end
function check_isfinite(::MatSqMut, x)
    if any(any.(!isfinite, x))
        i = findall(!isfinite, convert_to_array(x))
        throw(NLsolve.IsFiniteException(i))
    end
end
function check_isfinite(::NotMixed, x)
    if any(!isfinite, x)
        i = findall(!isfinite, x)
        throw(IsFiniteException(i))
    end
end


"""
    determinant(A)

Same as LinearAlgebra.det(A), works well with
mixed types.

    Examples, comparison with LinearAlgebra.det

```
julia> A = [1 2 3; 4 5 6; 7 8 10]m
3×3 Matrix{Quantity{Int64,  ᴸ, FreeUnits{(m,),  ᴸ, nothing}}}:
 1  2   3
 4  5   6
 7  8  10

julia> @time determinant(A)
  0.000014 seconds (42 allocations: 1.281 KiB)
-3m³

julia> @time LinearAlgebra.det(A)
  0.000029 seconds (73 allocations: 2.062 KiB)
-2.9999999999999982m³

julia> B = [1 1cm; -1cm^-1 1]
2×2 Matrix{Quantity{Int64}}:
                1  1cm
 -1cm⁻¹              1

julia> @time determinant(B)
0.000010 seconds (2 allocations: 80 bytes)
2

julia> LinearAlgebra.det(B)
ERROR: ArgumentError: cannot reinterpret `Quantity{Int64}` as `Int64`, type `Quantity{Int64}` is
not a bits type

julia> A3u = [1kg 2kg∙cm 3s; 4s 5cm∙s 6s²∙kg⁻¹; 7cm⁻¹ 8 0s∙kg⁻¹∙cm⁻¹]
3×3 Matrix{Quantity{Int64}}:
   1kg  2kg∙cm     3s
    4s   5cm∙s     6s²∙kg⁻¹
 7cm⁻¹   8        10s∙kg⁻¹∙cm⁻¹

 julia> @time determinant(A3u)
 0.000022 seconds (32 allocations: 1.125 KiB)
-3s²

julia> A3u_inco = [1kg 2kg∙cm 3s*cm; 4s 5cm∙s 6s²∙kg⁻¹; 7cm⁻¹ 8 9s∙kg⁻¹∙cm⁻¹]
1kg  2kg∙cm     3cm∙s
4s   5cm∙s     6s²∙kg⁻¹
7cm⁻¹   8        10s∙kg⁻¹∙cm⁻¹

julia> determinant(A3u_inco)
ERROR: DimensionMismatch: in determinant, Δ = 1 * 3cm∙s * -3s = -9cm∙s²    incompatible with accum = 9s²
 ...
```
"""
function determinant(A::AbstractArray)
    r, n = size(A)
    @assert r == n "r = $r, n = $n not square"
    if n == 1
        first(A)
    else
        accum = first(A) * determinant(@view A[2:n, 2:n])
        rngn = 1:n
        for j in 2:n
            entry = A[1, j]
            sig = 1 - 2 * iseven(j)
            cofactor = determinant(@view A[2:end, rngn .!= j])
            Δ = sig * entry *  cofactor
            if dimension(Δ) === dimension(accum)
                accum += Δ
            else
                msg = "in determinant, Δ = $sig * $entry * $cofactor = $Δ is incompatible with accum = $accum"
                throw( DimensionMismatch(msg))
            end
        end
        accum
    end
end


"""
    determinant_dimension(A)

    --> dimension (length, time, mass/ luminance...) of the determinant

If dimensions of the elements are incompatible
    --> `Dimensions{(Dimension{Missing}(1//1),)}`

...where determinant(A) would return an informative error message.

    Examples

```
julia> determinant_dimension([1 2 3; 4 5 6; 7 8 10])
NoDims

julia> A = [1 2 3; 4 5 6; 7 8 10]m
3×3 Matrix{Quantity{Int64,  ᴸ, FreeUnits{(m,),  ᴸ, nothing}}}:
 1  2   3
 4  5   6
 7  8  10

julia> @time determinant_dimension(A)
  0.000010 seconds (13 allocations: 848 bytes)
  ᴸ³

julia> B = [1 1cm; -1cm^-1 1]
2×2 Matrix{Quantity{Int64}}:
                1  1cm
 -1cm⁻¹              1

julia> @time determinant_dimension(B)
julia> @time determinant_dimension(B)
  0.000012 seconds (1 allocation: 64 bytes)
NoDims

julia> A3u = [1kg 2kg∙cm 3s; 4s 5cm∙s 6s²∙kg⁻¹; 7cm⁻¹ 8 0s∙kg⁻¹∙cm⁻¹]
3×3 Matrix{Quantity{Int64}}:
   1kg  2kg∙cm     3s
    4s   5cm∙s     6s²∙kg⁻¹
 7cm⁻¹   8        10s∙kg⁻¹∙cm⁻¹

 julia> @time determinant_dimension(A3u)
 0.000021 seconds (13 allocations: 848 bytes)
ᵀ²

julia> A3u_inco = [1kg 2kg∙cm 3s*cm; 4s 5cm∙s 6s²∙kg⁻¹; 7cm⁻¹ 8 9s∙kg⁻¹∙cm⁻¹]
1kg  2kg∙cm     3cm∙s
4s   5cm∙s     6s²∙kg⁻¹
7cm⁻¹   8        10s∙kg⁻¹∙cm⁻¹

julia> @time determinant_dimension(A3u_inco)
  0.000015 seconds (13 allocations: 848 bytes)
Dimensions{(Dimension{Missing}(1//1),)}
```
"""
function determinant_dimension(A)
    # Variable names are kept from `determinant_A`
    r, n = size(A)
    @assert r == n "r = $r, n = $n not square"
    if n == 1
        dimension(typeof(first(A)))
    else
        accum = dimension(first(A)) * determinant_dimension(@view A[2:n, 2:n])
        rngn = 1:n
        for j in 2:n
            entry = dimension(A[1, j])
            # Sign is irrelevant
            cofactor = determinant_dimension(@view A[2:end, rngn .!= j])
            Δ = entry * cofactor
            if Δ === accum
                # Addition won't change these compatible dimensions
            else
                return Dimensions{(Dimension{Missing}(1//1),)}
            end
        end
       accum
    end
end
