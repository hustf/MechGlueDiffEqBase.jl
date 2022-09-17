using NLsolve: AbstractSolverCache
"""
    LenNTRCache <: AbstractSolverCache

    Type lenient NewtonTrustRegionCache. Fieldnames:

    x::Tx         # Current point
    xold::Tx      # Old point
    r::TF         # Current residual
    r_predict::TF # Predicted residual
    p             # Step, dimensionless
    p_c::Tx       # Cauchy point, dimensionless
    pi            # Gauss-Newton step, dimensionless
    d             # Scaling vector, dimensionless
"""
struct LenNTRCache{Tx, TF, T} <: AbstractSolverCache
    x::Tx
    xold::Tx
    r::TF
    r_predict::TF
    p::T
    p_c::T
    pi::T
    d::T
end
function LenNTRCache(df)
    x = copy(df.x_f)
    xold = copy(x)
    r = copy(df.F)
    r_predict = copy(r)
    p = ustrip(copy(x))
    p_c = ustrip(copy(x))
    pi = ustrip(copy(x))
    d = ustrip(copy(x))
    LenNTRCache(x, xold, r, r_predict, p, p_c, pi, d)
end

function trust_region_(df::OnceDifferentiable,
    initial_x::AbstractArray{Tx},
    xtol::T,
    ftol::T,
    iterations::Integer,
    store_trace::Bool,
    show_trace::Bool,
    extended_trace::Bool,
    factor::Real,
    autoscale::Bool,
    cache = LenNTRCache(df)) where {Tx, T}



    copyto!(cache.x, initial_x)
    value_jacobian!!(df, cache.x)
    cache.r .= NLsolve.value(df)
    @debug "trust_region_:54 LenNTRCache evaluated at " string(initial_x) string(df.F) string(df.DF) string(cache.r) string(cache.d) Tx T maxlog = 2

    NLsolve.check_isfinite(cache.r)

    it = 0
    x_converged, f_converged = assess_convergence(ustrip(initial_x), ustrip(cache.xold), ustrip(NLsolve.value(df)), NaN, ftol)

    stopped = any(isnan, cache.x) || any(isnan, NLsolve.value(df)) ? true : false

    converged = x_converged || f_converged

    delta = convert(real(T), NaN)
    rho = convert(real(T), NaN)
    @debug "trust_region_:68" converged delta rho maxlog = 2
    if converged
        tr = NLsolve.SolverTrace()
        name = "Trust-region with dogleg"
        if autoscale
            name *= " and autoscaling"
        end
        return SolverResults(name,
            initial_x, copy(cache.x), norm(cache.r, Inf),
            it, x_converged, xtol, f_converged, ftol, tr,
            first(df.f_calls), first(df.df_calls))
    end

    tr = NLsolve.SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    NLsolve.@trustregiontrace convert(real(T), NaN)
    nn = length(cache.x)
    if autoscale
        J = NLsolve.jacobian(df)
        for j = 1:nn
            cache.d[j] = norm(view(ustrip(J), :, j))
            @debug "trust_region_:88" it j cache.d[j] string(J)
            if cache.d[j] == zero(cache.d[j])
                cache.d[j] = one(cache.d[j])
            end
        end
    else
        throw("hardly yet")
        # oneunit.(ArrayPartition(4.338005808987156kg, 1.4s))

        fill!(cache.d, one(real(T)))
    end
    @debug "trustregion_:100" string(cache.d) string(cache.x)
    delta = factor * NLsolve.wnorm(cache.d ./ oneunit.(cache.d), cache.x ./ oneunit.(cache.x))
    @debug "trustregion_" it T delta
    if delta == zero(delta)
        delta = factor
    end

    eta = convert(real(T), 1e-4)

    while !stopped && !converged && it < iterations
        it += 1

        @debug "trustregion_:115" it T delta string(cache.p) string(cache.p_c) string(cache.r) string(cache.d)
        # Compute proposed iteration step
        dogleg_dimensional!(cache.p, cache.p_c, cache.pi, cache.r, cache.d, NLsolve.jacobian(df), delta)
        copyto!(cache.xold, cache.x)
        cache.x .+= cache.p
        NLsolve.value!(df, cache.x)

        # Ratio of actual to predicted reduction (equation 11.47 in N&W)
        mul!(vec(cache.r_predict), NLsolve.jacobian(df), vec(cache.p))
        cache.r_predict .+= cache.r

        rho = (sum(abs2, cache.r) - sum(abs2, NLsolve.value(df))) / (sum(abs2, cache.r) - sum(abs2, cache.r_predict))
        @debug "trustregion_" it T rho eta
        if rho > eta
            # Successful iteration
            cache.r .= NLsolve.value(df)
            NLsolve.jacobian!(df, cache.x)

            # Update scaling vector
            if autoscale
                for j = 1:nn
                    cache.d[j] = max(convert(real(T), 0.1) * real(cache.d[j]), norm(view(NLsolve.jacobian(df), :, j)))
                end
            end

            x_converged, f_converged = assess_convergence(ustrip(cache.x), ustrip(cache.xold), ustrip(cache.r), xtol, ftol)
            converged = x_converged || f_converged
        else
            cache.x .-= cache.p
            x_converged, converged = false, false
        end

        NLsolve.@trustregiontrace euclidean(cache.x, cache.xold)

        # Update size of trust region
        if rho < 0.1
            delta = delta/2
        elseif rho >= 0.9
            delta = 2 * NLsolve.wnorm(cache.d, cache.p)
        elseif rho >= 0.5
            delta = max(delta, 2 * NLsolve.wnorm(cache.d, cache.p))
        end
        stopped = any(isnan, cache.x) || any(isnan, NLsolve.value(df)) ? true : false
    end

    name = "Trust-region with dogleg"
    if autoscale
        name *= " and autoscaling"
    end
    return NLsolve.SolverResults(name,
        initial_x, copy(cache.x), maximum(abs, cache.r),
        it, x_converged, xtol, f_converged, ftol, tr,
        first(df.f_calls), first(df.df_calls))
    true
end
function trust_region(df::OnceDifferentiable,
    initial_x::RW(N),
    xtol::Real,
    ftol::Real,
    iterations::Integer,
    store_trace::Bool,
    show_trace::Bool,
    extended_trace::Bool,
    factor::Real,
    autoscale::Bool) where N
    cache = LenNTRCache(df)
    @debug "trust_region:187 LenNTRCache" xtol ftol string(initial_x) iterations factor N maxlog = 2
    trust_region_(df, initial_x, xtol, ftol, iterations, store_trace, show_trace, extended_trace,
        convert(numtype(xtol), factor), autoscale, cache)
end


function dogleg_dimensional!(p, p_c, p_i,
                 r, d, J, delta::Real)
    T = eltype(d)
    @debug "dogleg_dimensional!:188" string(p) string(p_c) string(p_i) string(r) string(d) string(J) delta maxlog = 2
    dimstep = J \ r
    p_i = ustrip(dimstep)
    #try
        copyto!(p_i, J \ vec(r)) # Gauss-Newton step
    #catch e
    #    if isa(e, LAPACKException) || isa(e, SingularException)
            # If Jacobian is singular, compute a least-squares solution to J*x+r = 0
    #        U, S, V = svd(convert(Matrix{T}, J)) # Convert to full matrix because sparse SVD not implemented as of Julia 0.3
    #        k = sum(S .> eps())
    #        mrinv = V * Matrix(Diagonal([1 ./ S[1:k]; zeros(eltype(S), length(S)-k)])) * U' # Moore-Penrose generalized inverse of J
    #        vecpi = vec(p_i)
    #        mul!(vecpi,mrinv,vec(r))
    #    else
    #        throw(e)
    #    end
    #end
    rmul!(p_i, -one(T))

    # Test if Gauss-Newton step is within the region
    if wnorm(d, p_i) <= delta
        copyto!(p, p_i)   # accepts equation 4.13 from N&W for this step
    else
        # For intermediate we will use the output array `p` as a buffer to hold
        # the gradient. To make it easy to remember which variable that array
        # is representing we make g an alias to p and use g when we want the
        # gradient

        # compute g = J'r ./ (d .^ 2)
        g = p
        mul!(vec(g), transpose(J), vec(r))
        g .= g ./ d.^2

        # compute Cauchy point
        p_c .= -wnorm(d, g)^2 / sum(abs2, J*vec(g)) .* g

        if wnorm(d, p_c) >= delta
            # Cauchy point is out of the region, take the largest step along
            # gradient direction
            rmul!(g, -delta/wnorm(d, g))

            # now we want to set p = g, but that is already true, so we're done

        else
            # from this point on we will only need p_i in the term p_i-p_c.
            # so we reuse the vector p_i by computing p_i = p_i - p_c and then
            # just so we aren't confused we name that p_diff
            p_i .-= p_c
            p_diff = p_i

            # Compute the optimal point on dogleg path
            b = 2 * wdot(d, p_c, d, p_diff)
            a = wnorm(d, p_diff)^2
            tau = (-b + sqrt(b^2 - 4a*(wnorm(d, p_c)^2 - delta^2)))/(2a)
            p_c .+= tau .* p_diff
            copyto!(p, p_c)
        end
    end
end
