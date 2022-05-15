###############################
# Differentiation of quantities
###############################

# The function signature in OrdinaryDiffEq.FiniteDiff is restrictive. "Real" excludes complex numbers,
# but that unfortunately excludes Quantity as well. A Quantity covering several types can be Real or Complex.
# compute_epsilon extends \FiniteDiff\src\epsilons.jl
@inline function compute_epsilon(::Val{:central}, x::T, relstep::Real, absstep::Quantity{T1, D, U}, dir=nothing) where {T<:Number, T1<:Real, D, U}
    max(relstep * abs(x), absstep)
end

@inline function compute_epsilon(::Val{:forward}, x::T, relstep::Real, absstep::Quantity{T1, D, U}, 
    dir = nothing) where {T<:Number, T1<:Real, D, U}
    max(relstep * oneunit(absstep), absstep)
end

@inline function compute_epsilon(::Val{:complex}, x::Quantity{T, D, U}, ::Union{Nothing,T1}=nothing, 
    ::Union{Nothing,Quantity{T, D, U}}=nothing, dir=nothing) where {T1<:Real, T<:Real, D, U}
    eps(Quantity{T, D, U})
end


# Extend Unitifu function for non-quantities
numtype(x::Type) = x


##############################
# (Updating) Jacobian matrices
# with mixed element types
##############################

# From FiniteDiff\src\derivatives.jl:4. The only change here is the default absstep argument. 
function finite_difference_derivative(
    f,
    x::T,
    fdtype=Val(:central),
    returntype=eltype(x),
    f_x=nothing;
    relstep=default_relstep(fdtype, T),
    absstep=relstep * oneunit(x),
    dir=true) where {T<:Quantity}

    fdtype isa Type && (fdtype = fdtype())
    epsilon = compute_epsilon(fdtype, x, relstep, absstep, dir)
    if fdtype==Val(:forward)
        return (f(x+epsilon) - f(x)) / epsilon
    elseif fdtype==Val(:central)
        return (f(x+epsilon) - f(x-epsilon)) / (2*epsilon)
    elseif fdtype==Val(:complex) && returntype<:Real
        return imag(f(x+im*epsilon)) / epsilon
    end
    fdtype_error(returntype)
end


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
            @debug "calculate_Ji_forward " vecx vecx1 epsilon i #x_save x1_save x_save x
            _vecx1 = Base.setindex(vecx1,x1_save+epsilon,i)
            _vecx = Base.setindex(vecx,x_save-epsilon,i)
            _x1 = reshape(_vecx1, axes(x))
            _x = reshape(_vecx, axes(x))
            vecfx1 = _vec(f(_x1))
            vecfx = _vec(f(_x))
            dx = (vecfx1-vecfx)/(2epsilon)
            @debug "calculate_Ji_forward " repr(vecfx) repr(vecfx1) epsilon i x_save _x1 x
            return dx
        end

        if jac_prototype isa Nothing && sparsity isa Nothing
            Jvec = map(calculate_Ji_central, 1:maximum(colorvec))
            J = ArrayPartition(map(ArrayPartition, zip(Jvec...))...)
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
        @debug "Untested" maxlog = 1
        fdtype_error(returntype)
    end
    J
end
