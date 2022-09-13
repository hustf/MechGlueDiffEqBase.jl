# From FiniteDiff\src\jacobians.jl:133
# Dispatches on mutable ArrayParitions, the only other change is
# default absstep, which needs to conform to the units of x.
function finite_difference_jacobian(f, x::ArrayPartition,
    fdtype::Val     = Val(:forward), 
    returntype = fdtype == Val(:complex) ? numtype(eltype(f(x))) : eltype(f(x)),
    f_in       = nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep = relstep .* oneunit.(x), # Units can vary between 'rows' in the ArrayPartition.
    colorvec = 1:length(x),
    sparsity = nothing,
    jac_prototype = nothing,
    dir=true) where N
    x_mutable = map(x-> [x], x)
    finite_difference_jacobian(f, x_mutable, fdtype, returntype, f_in; relstep, absstep, colorvec, sparsity, jac_prototype, dir)
end
function finite_difference_jacobian(f, x::RW(N),
    fdtype::Val     = Val(:forward), 
    returntype = fdtype == Val(:complex) ? numtype(eltype(f(x))) : eltype(f(x)),
    f_in       = nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep = relstep .* oneunit.(x), # Units can vary between 'rows' in the ArrayPartition.
    colorvec = 1:length(x),
    sparsity = nothing,
    jac_prototype = nothing,
    dir=true) where N
    if f_in isa Nothing
        fx = f(x)
    else
        @debug "Untested" maxlog = 1
        fx = f_in
    end
    if fdtype == Val(:complex)
        xcomp = complex.(zero(x))
        cache = JacobianCache(xcomp, f(xcomp), fdtype, returntype)
    else
        cache = JacobianCache(x, fx, fdtype, returntype)
    end
    # Targeting method defined below, which is extending 
    # FiniteDiff\src\jacobians.jl:155.
    @debug "finite_difference_jaco"
    finite_difference_jacobian(f, x, cache, f_in;
        relstep, absstep, colorvec, sparsity, jac_prototype, dir)
end

function finite_difference_jacobian(
    f,
    x::ArrayPartition,
    cache::JacobianCache{T1,T2,T3,cType,sType,fdtype,returntype},
    f_in=nothing;
    relstep=default_relstep(fdtype, eltype(x)),
    absstep=relstep,
    colorvec = cache.colorvec,
    sparsity = cache.sparsity,
    jac_prototype = nothing,
    dir=true) where {T1,T2,T3,cType,sType,fdtype,returntype}

    x1, fx, fx1 = cache.x1, cache.fx, cache.fx1

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

    J = jac_prototype isa Nothing ? (sparsity isa Nothing ? jacobian_prototype_zero(x, vecfx) : zeros(eltype(x),size(sparsity))) : zero(jac_prototype)
    nrows = length(J.x)
    ncols = length(J.x[1])
     # TODO Indiexing with indices obtained from length, size etc. is discouraged. Use eachindex or axes instead.
    if !(sparsity isa Nothing)
        rows_index, cols_index = ArrayInterface.findstructralnz(sparsity)
        rows_index = [rows_index[i] for i in 1:length(rows_index)]
        cols_index = [cols_index[i] for i in 1:length(cols_index)]
    end

    if fdtype == Val(:forward)
        "Vary the ith element of vecx. The result is the ith column in the Jacobian."
        function calculate_Ji_forward(i)
            x_save = ArrayInterface.allowed_getindex(vecx, i)
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
        if jac_prototype isa Nothing && sparsity isa Nothing
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
                if sparsity isa Nothing
                    dx = calculate_Ji_forward(color_i)
                    J = J + _make_Ji(J, eltype(x), dx, color_i, nrows, ncols)
                else
                    tmp = norm(vecx .* (colorvec .== color_i))
                    epsilon = compute_epsilon(Val(:forward), sqrt(tmp), relstep, absstep, dir)
                    _vecx = @. vecx + epsilon * (colorvec == color_i)
                    _x = reshape(_vecx, axes(x))
                    vecfx1 = _vec(f(_x))
                    dx = (vecfx1-vecfx)/epsilon
                    Ji = _make_Ji(J,rows_index,cols_index,dx,colorvec,color_i,nrows,ncols)
                    J = J + Ji
                end
            end
        end
    elseif fdtype == Val(:central)
        function calculate_Ji_central(i)
            x1_save = ArrayInterface.allowed_getindex(vecx1,i)
            x_save = ArrayInterface.allowed_getindex(vecx,i)
            epsilon = compute_epsilon(Val(:central), x_save, relstep, absstep[i], dir)
            @debug "calculate_Ji_central" repr(vecx) repr(vecx1) epsilon i #x_save x1_save x_save x
            #_vecx1 = Base.setindex(vecx1,x1_save+epsilon,i)
            #_vecx = Base.setindex(vecx,x_save-epsilon,i)
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

        if jac_prototype isa Nothing && sparsity isa Nothing
            Jvec = map(calculate_Ji_central, 1:maximum(colorvec))
            for (j, column) in zip(1:ncols, Jvec)
                for (i, el) in zip(1:nrows, column)
                    J[i, j] = el
                end
            end
        else
            @debug "Untested" maxlog = 1
            @inbounds for color_i ∈ 1:maximum(colorvec)
                if sparsity isa Nothing
                    @debug "Untested" maxlog = 1
                    dx = calculate_Ji_central(color_i)
                    J = J + _make_Ji(J, eltype(x), dx, color_i, nrows, ncols)
                else
                    @debug "Untested" maxlog = 1
                    tmp = norm(vecx1 .* (colorvec .== color_i))
                    epsilon = compute_epsilon(Val(:forward), sqrt(tmp), relstep, absstep, dir)
                    _vecx1 = @. vecx1 + epsilon * (colorvec == color_i)
                    _vecx = @. vecx - epsilon * (colorvec == color_i)
                    _x1 = reshape(_vecx1, axes(x))
                    _x = reshape(_vecx, axes(x))
                    vecfx1 = _vec(f(_x1))
                    vecfx = _vec(f(_x))
                    dx = (vecfx1-vecfx)/(2epsilon)
                    Ji = _make_Ji(J,rows_index,cols_index,dx,colorvec,color_i,nrows,ncols)
                    J = J + Ji
                end
            end
        end
    elseif fdtype == Val(:complex) && returntype <: Real
        function calculate_Ji_complex(i)
            _vecx = complex.(vecx)
            x_save = ArrayInterface.allowed_getindex(vecx, i)
            epsilon = compute_epsilon(Val(:complex), x_save, relstep, absstep[i], dir)
            su = x_save + im * oneunit(x_save) * epsilon
            @debug "calculate_Ji_complex"  vecfx epsilon i x_save x su  returntype _vecx
            setindex!(_vecx, su, i)
            _x = reshape(_vecx, axes(x))
            vecfx = MechGlueDiffEqBase._vec(f(_x))
            @debug "calculate_Ji_complex"  repr(vecfx) 
            imag(vecfx) / (epsilon * oneunit(x_save))
        end

        if jac_prototype isa Nothing && sparsity isa Nothing
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
                if sparsity isa Nothing
                    dx = calculate_Ji_complex(color_i)
                    J = J + _make_Ji(J, eltype(x), dx, color_i, nrows, ncols)
                else
                    @debug "Untested" maxlog = 1
                    _vecx = @. vecx + im * epsilon * (colorvec == color_i)
                    _x = reshape(_vecx, axes(x))
                    vecfx = _vec(f(_x))
                    dx = imag(vecfx)/epsilon
                    Ji = _make_Ji(J,rows_index,cols_index,dx,colorvec,color_i,nrows,ncols)
                    J = J + Ji
                end
            end
        end
    else
        @debug "finite_difference_jacobian " fdtype returntype maxlog = 1
        fdtype_error(fdtype) # returntype
    end
    J
end
