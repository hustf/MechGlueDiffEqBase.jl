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
    @debug "trust_region_:55 LenNTRCache evaluated at " string(initial_x) string(df.F) string(df.DF) string(cache.r) string(cache.d) Tx T maxlog = 2

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
        return SolverResultsDimensional(name,     # method::String
            initial_x,                            # initial_x::I
            copy(cache.x),                        # zero::Z
            maximum(x-> abs(ustrip(x)), cache.r), # residual_norm::rT
            it,                                   # iterations::Int
            x_converged,                          # x_converged::Bool
            xtol,                                 # xtol::rT
            f_converged,                          # f_converged::Bool
            ftol,                                 # ftol::rT
            tr,                                   # trace::SolverTrace
            first(df.f_calls),                    # f_calls::Int
            first(df.df_calls))                   # g_calls::Int  
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
        throw("autoscale = false not implemented and tested yet")
        # oneunit.(ArrayPartition(4.338005808987156kg, 1.4s))
        fill!(cache.d, one(real(T)))
    end
    @debug "trustregion_:107" string(cache.d) string(cache.x)
    delta = factor * NLsolve.wnorm(cache.d ./ oneunit.(cache.d), cache.x ./ oneunit.(cache.x))
    @debug "trustregion_" it T delta
    if delta == zero(delta)
        delta = factor
    end

    eta = convert(real(T), 1e-4)

    while !stopped && !converged && it < iterations
        it += 1
        @debug "trustregion_:119" it T delta string(cache.p) string(cache.p_c) string(cache.r) string(cache.d)
        # Compute proposed iteration step
        dogleg_dimensional!(cache.p, cache.p_c, cache.pi, cache.r, cache.d, NLsolve.jacobian(df), delta)
        copyto!(cache.xold, cache.x)
        cache.x .+= cache.p .* unit.(cache.x)
        @debug "trustregion_:123" it string(cache.x) maxlog = 10

        NLsolve.value!(df, cache.x)
        @debug "trustregion_:126" string(cache.r_predict) string(NLsolve.jacobian(df)) string(cache.p) maxlog = 1
        # Ratio of actual to predicted reduction (equation 11.47 in N&W)
        mul!(cache.r_predict, NLsolve.jacobian(df), cache.p .* unit.(cache.x))
        @debug "trustregion_:129" string(cache.r_predict) string(cache.r)
        cache.r_predict .+= cache.r
        rho = (sum(UNITLESS_ABS2, cache.r) - sum(UNITLESS_ABS2, NLsolve.value(df))) / (sum(UNITLESS_ABS2, cache.r) - sum(UNITLESS_ABS2, cache.r_predict))
        @debug "trustregion_:132" it T rho eta
        if rho > eta
            # Successful iteration
            @debug "trustregion_:135 success" 
            cache.r .= NLsolve.value(df)
            @debug "trustregion_:137 update jacobian" 
            NLsolve.jacobian!(df, cache.x)
            @debug "trustregion_:139 update scaling " 
            # Update scaling vector
            if autoscale
                for j = 1:nn
                    cache.d[j] = max(convert(real(T), 0.1) * real(cache.d[j]), ODE_DEFAULT_NORM(view(NLsolve.jacobian(df), :, j), 2))
                    @debug "trustregion_:144 norm col $j " ODE_DEFAULT_NORM(view(NLsolve.jacobian(df), :, j), 0)
                end
            end
            @debug "trustregion_:147"
            x_converged, f_converged = assess_convergence(ustrip(cache.x), ustrip(cache.xold), ustrip(cache.r), xtol, ftol)
            @debug "trustregion_:149"
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
    return SolverResultsDimensional(name,            # method::String
            initial_x,                            # initial_x::I
            copy(cache.x),                        # zero::Z
            maximum(x-> abs(ustrip(x)), cache.r), # residual_norm::rT
            it,                                   # iterations::Int
            x_converged,                          # x_converged::Bool
            xtol,                                 # xtol::rT
            f_converged,                          # f_converged::Bool
            ftol,                                 # ftol::rT
            tr,                                   # trace::SolverTrace
            first(df.f_calls),                    # f_calls::Int
            first(df.df_calls))                   # g_calls::Int  
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
    @debug "trust_region:197 LenNTRCache" xtol ftol string(initial_x) iterations factor N maxlog = 2
    trust_region_(df, initial_x, xtol, ftol, iterations, store_trace, show_trace, extended_trace,
        convert(numtype(xtol), factor), autoscale, cache)
end


function dogleg_dimensional!(p, p_c, p_i,
                 r, d, J, delta::Real)
    T = eltype(d)
    @debug "dogleg_dimensional!:206" string(p) string(p_c) string(p_i) string(r) string(d) string(J) delta maxlog = 2
    # TODO do this in-place. Consider propagate errors like commented below.
    #try
        copyto!(p_i, ustrip.(J \ r))# Gauss-Newton step
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
    @debug "dogleg_dimensional!:222" string(p_i) T maxlog = 5
    NLsolve.rmul!(p_i, -one(T))
    # Test if Gauss-Newton step is within the region
    if NLsolve.wnorm(d, p_i) <= delta
        copyto!(p, p_i)   # accepts equation 4.13 from N&W for this step
    else
        # For intermediate we will use the output array `p` as a buffer to hold
        # the gradient. To make it easy to remember which variable that array
        # is representing we make g an alias to p and use g when we want the
        # gradient
        # compute g = J'r ./ (d .^ 2)
        g = p
        @debug "dogleg_dimensional!:234" string(g) string(J) string(r)
        g .= ustrip.(J \ r)
        @debug "dogleg_dimensional!:236" string(g)
        g .= g ./ d.^2
        @debug "dogleg_dimensional!:238" string(g) string(d)
        # compute Cauchy point
        p_c .= -NLsolve.wnorm(d, g)^2 / UNITLESS_ABS2(ustrip(J) * g) .* g
        @debug "dogleg_dimensional!:241" string(p_c) delta NLsolve.wnorm(d, p_c)
        if NLsolve.wnorm(d, p_c) >= delta
            # Cauchy point is out of the region, take the largest step along
            # gradient direction
            NLsolve.rmul!(g, -delta/NLsolve.wnorm(d, g))

            # now we want to set p = g, but that is already true, so we're done

        else
            # from this point on we will only need p_i in the term p_i-p_c.
            # so we reuse the vector p_i by computing p_i = p_i - p_c and then
            # just so we aren't confused we name that p_diff
            p_i .-= p_c
            p_diff = p_i

            # Compute the optimal point on dogleg path
            b = 2 * NLsolve.wdot(d, p_c, d, p_diff)
            a = NLsolve.wnorm(d, p_diff)^2
            tau = (-b + sqrt(b^2 - 4a*(NLsolve.wnorm(d, p_c)^2 - delta^2)))/(2a)
            p_c .+= tau .* p_diff
            copyto!(p, p_c)
        end
    end
end
