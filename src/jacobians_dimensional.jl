# Extends FiniteDiff\src\jacobians.jl:300.
# Sets absstep to a vector with the same types as elements of x.
# Constructs types of fx from J and x.
# Constructs a JacobianCache.
function finite_difference_jacobian!(J::MixedCandidate,
    f,
    x,
    fdtype::Val = Val(:forward),
    returntype = eltype(x),
    f_in       = nothing;
    relstep = default_relstep(fdtype, eltype(x)),
    absstep = relstep .* oneunit.(x), # A (mixed) vector
    colorvec = 1:length(x),
    sparsity = nothing)
    #
    @assert is_square_matrix_mutable(J)
    @assert size(J)[2] == length(x)
    @assert size(absstep)[1] == size(x)[1]
    @assert !(fdtype isa Type)
    @debug "finite_difference_jacobian!:20" fdtype returntype relstep repr(absstep) maxlog=2
    if f_in isa Nothing
        fx = J[:, 1] * oneunit(x[1]) # Get the types and dimension right
        @debug "finite_difference_jacobian!:23" string(x) string(fx) maxlog=2
        f(fx, x)          # Get the values of fx right. Also checks if J is dimensionally correct.
        if fdtype == Val(:complex) && returntype <: Quantity
            # Quantities are not a subtype of Real. Fool the outside constructor:
            _x = zero.(complex.(x))
            _fx = zero.(complex.(fx))
            c = JacobianCache(_x, _fx, fdtype, numtype(returntype))
            # Copy through the inner constructor, now with modified returntype and numeric type
            cache = JacobianCache{typeof(c.x1), typeof(c.x2), typeof(c.fx), typeof(c.fx1), typeof(c.colorvec), typeof(c.sparsity), fdtype, returntype}(c.x1, c.x2, c.fx, c.fx1, c.colorvec, c.sparsity)
        else
            cache = JacobianCache(x, fx, fdtype, returntype)
        end
    else
        cache = JacobianCache(x, f_in, fdtype, returntype)
    end
    @debug "finite_difference_jacobian!:36" repr(cache.x1) repr(cache.fx) maxlog=2
    finite_difference_jacobian!(J, f, x, cache, cache.fx; relstep, absstep, colorvec, sparsity)
end


# Extends FiniteDiff\src\jacobians.jl:344.
# Sets absstep to a vector with the same types as elements of x.
# Excludes sparse matrix functionality.
function finite_difference_jacobian!(
    J::MixedCandidate,
    f,
    x,
    cache::JacobianCache{T1,T2,T3,T4,cType,sType,fdtype,returntype},
    f_in = nothing;
    relstep = default_relstep(fdtype, eltype(x)),
    absstep = relstep .* oneunit.(x), # A mixed vector,
    colorvec = cache.colorvec,
    sparsity = cache.sparsity,
    dir = true) where {T1,T2,T3,T4,cType,sType,fdtype,returntype}
    @assert is_square_matrix_mutable(J)
    @assert size(absstep)[1] == size(x)[1]
    @assert sparsity isa Nothing
    @debug "finite_difference_jacobian!:58" fdtype returntype sparsity maxlog=2
    m, n = size(J)
    _color = reshape(colorvec, axes(x)...)

    x1, x2, fx, fx1 = cache.x1, cache.x2, cache.fx, cache.fx1
    copyto!(x1, x)
    vfx = _vec(fx)
    if fdtype == Val(:forward)
        vfx1 = _vec(fx1)

        if f_in isa Nothing
            f(fx, x)
            vfx = _vec(fx)
        else
            vfx = _vec(f_in)
        end

        @inbounds for color_i ∈ 1:maximum(colorvec) # Consider broadcast
            x1_save = ArrayInterfaceCore.allowed_getindex(x1,color_i)
            absstep_save = ArrayInterfaceCore.allowed_getindex(absstep,color_i)
            epsilon = compute_epsilon.(Val(:forward), x1_save, relstep, absstep_save, dir)
            @debug "finite_difference_jacobian:79" string(x1) string(x1_save) string(epsilon) color_i maxlog = 10
            ArrayInterfaceCore.allowed_setindex!(x1, x1_save + epsilon, color_i)
            f(fx1, x1)
            # J is dense, so either it is truly dense or this is the
            # compressed form of the coloring, so write into it.
            @. J[:,color_i] = (vfx1 - vfx) / epsilon
            # Now return x1 back to its original value
            ArrayInterfaceCore.allowed_setindex!(x1, x1_save, color_i)
        end #for ends here
    elseif fdtype == Val(:central) # Consider broadcast
        vfx1 = _vec(fx1)
        @inbounds for color_i ∈ 1:maximum(colorvec)
            x_save = ArrayInterfaceCore.allowed_getindex(x, color_i)
            absstep_save = ArrayInterfaceCore.allowed_getindex(absstep, color_i)
            epsilon = compute_epsilon.(Val(:central), x_save, relstep, absstep_save, dir)
            @debug "finite_difference_jacobian:94" string(x1) string(x_save) string(epsilon) color_i maxlog = 10
            ArrayInterfaceCore.allowed_setindex!(x1, x_save + epsilon, color_i)
            f(fx1, x1)
            ArrayInterfaceCore.allowed_setindex!(x1, x_save - epsilon, color_i)
            f(fx, x1)
            @. J[:,color_i] = (vfx1 - vfx) / 2epsilon
            ArrayInterfaceCore.allowed_setindex!(x1, x_save, color_i)
        end
    elseif fdtype == Val(:complex) && numtype(returntype) <:Real # Consider broadcast
        epsilon = eps(eltype(x))
        @inbounds for color_i ∈ 1:maximum(colorvec)
            x1_save = ArrayInterfaceCore.allowed_getindex(x1, color_i)
            @debug "finite_difference_jacobian:106" string(x1) string(x1_save) string(epsilon) color_i maxlog = 10
            _x = x1_save + im*epsilon * oneunit(x1_save)
            @debug "finite_difference_jacobian:108" _x typeof(_x) typeof(x1[color_i]) maxlog = 10
            x1[color_i] = _x
            @debug "finite_difference_jacobian:110" maxlog = 10
            ArrayInterfaceCore.allowed_setindex!(x1, x1_save + im*epsilon * oneunit(x1_save), color_i)
            @debug "finite_difference_jacobian:112" maxlog = 10
            f(fx,x1)
            @debug "finite_difference_jacobian:116" string(imag(vfx)) string(epsilon) string(J) color_i maxlog = 10
            @. J[:,color_i] = imag(vfx) / (epsilon * oneunit(x1_save))
            ArrayInterfaceCore.allowed_setindex!(x1, x1_save,color_i)
        end
    else
        fdtype_error(returntype)
    end
    @debug "finite_difference_jacobian:122" string(J) maxlog = 20
    nothing
end

# Extends FiniteDiff\src\jacobians.jl:137.
# Sets absstep to a vector with the same types as elements of x.
# Constructs types of fx by evaluating f(x), assuming f is in-place.
# Constructs a JacobianCache.
function finite_difference_jacobian(f, x::MixedCandidate,
    fdtype::Val     = Val(:forward),
    returntype = eltype(x),
    f_in       = nothing;
    relstep = default_relstep(fdtype, eltype(x)),
    absstep = relstep .* oneunit.(x), # A mixed vector
    colorvec = 1:length(x),
    sparsity = nothing,
    jac_prototype = nothing,
    dir = true)
    #
    @assert is_vector_mutable_stable(x)
    @assert size(absstep)[1] == size(x)[1]
    @debug "finite_difference_jacobian:135" string(x) fdtype returntype maxlog = 2
    if f_in isa Nothing
        fx = f(x)
    else
        fx = f_in
    end
    cache = JacobianCache(x, fx, fdtype, returntype)
    @debug "finite_difference_jacobian:142" typeof(cache) maxlog = 2
    finite_difference_jacobian(f, x, cache, fx; relstep, absstep, colorvec, sparsity, jac_prototype, dir)
end
# Extends FiniteDiff\src\jacobians.jl:159.
# Asserts absstep is a mixed vector.
# Disregards sparse
function finite_difference_jacobian(
    f,
    x::MixedCandidate,
    cache::JacobianCache{T1,T2,T3,T4,cType,sType,fdtype,returntype},
    f_in = nothing;
    relstep = default_relstep(fdtype, eltype(x)),
    absstep = relstep .* oneunit.(x), # A mixed vector
    colorvec = cache.colorvec,
    sparsity = cache.sparsity,
    jac_prototype = nothing,
    dir=true) where {T1,T2,T3,T4,cType,sType,fdtype,returntype}
    #
    @assert is_vector_mutable_stable(absstep)
    @assert sparsity isa Nothing
    @debug "finite_difference_jacobian:162" string(x) fdtype returntype string(absstep) f_in maxlog = 2
    x1, fx, fx1 = cache.x1, cache.fx, cache.fx1
    #
    # Rest of function is re-used from earlier version...
    if !(f_in isa Nothing)
        vecfx = MechGlueDiffEqBase._vec(f_in)
    elseif fdtype == Val(:forward)
        vecfx = MechGlueDiffEqBase._vec(f(x))
    elseif fdtype == Val(:complex) && returntype <: Real
        vecfx = real(fx)
    else
        vecfx = MechGlueDiffEqBase._vec(fx)
    end
    vecx = MechGlueDiffEqBase._vec(x)
    vecx1 = MechGlueDiffEqBase._vec(x1)

    J = jac_prototype isa Nothing ? jacobian_prototype_zero(x, vecfx)  : zero(jac_prototype)

    nrows = length(J.x)
    ncols = length(J.x[1])
    if fdtype == Val(:forward)
        "Vary the ith element of vecx. The result is the ith column in the Jacobian."
        function calculate_Ji_forward(i)
            x_save = ArrayInterfaceCore.allowed_getindex(vecx, i)
            epsilon = compute_epsilon(Val(:forward), x_save, relstep, absstep[i], dir)
            su = x_save + epsilon
            setindex!(vecx1, su, i)
            _x1 = reshape(vecx1, axes(x))
            vecfx1 = MechGlueDiffEqBase._vec(f(_x1))
            @debug "calculate_Ji_forward " repr(vecfx) repr(vecfx1) epsilon i x_save _x1 x
            d = (vecfx1-vecfx) / epsilon
            setindex!(vecx1, x_save, i)
            d
        end
        if jac_prototype isa Nothing
            Jvec = map(calculate_Ji_forward, 1:maximum(colorvec))
            for (j, column) in zip(1:ncols, Jvec)
                for (i, el) in zip(1:nrows, column)
                    J[i, j] = el
                end
            end
        else
            throw("Unexpected?")
            @debug "Untested" maxlog = 1
            @inbounds for color_i ∈ 1:maximum(colorvec)
                dx = calculate_Ji_forward(color_i)
                J = J + _make_Ji(J, eltype(x), dx, color_i, nrows, ncols)
            end
        end
    elseif fdtype == Val(:central)
        function calculate_Ji_central(i)
            x1_save = ArrayInterfaceCore.allowed_getindex(vecx1,i)
            x_save = ArrayInterfaceCore.allowed_getindex(vecx,i)
            epsilon = compute_epsilon(Val(:central), x_save, relstep, absstep[i], dir)
            @debug "calculate_Ji_central" repr(vecx) repr(vecx1) epsilon i #x_save x1_save x_save x
            setindex!(vecx1, x1_save+epsilon,i)
            setindex!(vecx, x_save-epsilon,i)
            _x1 = reshape(vecx1, axes(x))
            _x = reshape(vecx, axes(x))
            vecfx1 = _vec(f(_x1))
            vecfx = _vec(f(_x))
            d = (vecfx1-vecfx)/(2epsilon)
            @debug "calculate_Ji_forward " repr(vecfx) repr(vecfx1) epsilon i x_save _x1 x
            setindex!(vecx1, x1_save, i)
            setindex!(vecx, x_save, i)
            d
        end

        if jac_prototype isa Nothing
            Jvec = map(calculate_Ji_central, 1:maximum(colorvec)) # Consider if broadcast may be more inferrable
            for (j, column) in zip(1:ncols, Jvec)
                for (i, el) in zip(1:nrows, column)
                    J[i, j] = el
                end
            end
        else
            @debug "Untested" maxlog = 1
            @inbounds for color_i ∈ 1:maximum(colorvec)
                @debug "Untested" maxlog = 1
                dx = calculate_Ji_central(color_i)
                J = J + _make_Ji(J, eltype(x), dx, color_i, nrows, ncols)
            end
        end
    elseif fdtype == Val(:complex) && returntype <: Real
        function calculate_Ji_complex(i)
            _vecx = complex.(vecx)
            x_save = ArrayInterfaceCore.allowed_getindex(vecx, i)
            epsilon = compute_epsilon(Val(:complex), x_save, relstep, absstep[i], dir)
            su = x_save + im * oneunit(x_save) * epsilon
            @debug "calculate_Ji_complex"  vecfx epsilon i x_save x su  returntype _vecx
            setindex!(_vecx, su, i)
            _x = reshape(_vecx, axes(x))
            vecfx = MechGlueDiffEqBase._vec(f(_x))
            @debug "calculate_Ji_complex"  repr(vecfx)
            imag(vecfx) / (epsilon * oneunit(x_save))
        end

        if jac_prototype isa Nothing
            Jvec = map(calculate_Ji_complex, 1:maximum(colorvec))
            #J = ArrayPartition(map(ArrayPartition, zip(Jvec...))...)
            for (j, column) in zip(1:ncols, Jvec)
                for (i, el) in zip(1:nrows, column)
                    J[i, j] = el
                end
            end
        else
            @debug "Untested" maxlog = 1
            @inbounds for color_i ∈ 1:maximum(colorvec)
            dx = calculate_Ji_complex(color_i)
            J = J + _make_Ji(J, eltype(x), dx, color_i, nrows, ncols)
            end
        end
    else
        @debug "finite_difference_jacobian:275" fdtype returntype maxlog = 1
        fdtype_error(fdtype)
    end
    J
end
