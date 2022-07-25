using NLsolve: AbstractSolverCache
"""
    LenNTRCache <: AbstractSolverCache
    
    Type lenient NewtonTrustRegionCache. Fieldnames:

    x::Tx         # Current point
    xold::Tx      # Old point
    r::TF         # Current residual
    r_predict::TF # Predicted residual
    p             # Step
    p_c::Tx       # Cauchy point
    pi            # Gauss-Newton step
    d             # Scaling vector
"""
struct LenNTRCache{Tx, TF} <: AbstractSolverCache
    x::Tx         
    xold::Tx      
    r::TF         
    r_predict::TF 
    p::Tx
    p_c::Tx
    pi
    d
end
function LenNTRCache(df)
    x = copy(df.x_f)     
    xold = copy(x)       
    r = copy(df.F)       
    r_predict = copy(r)
    p = copy(x)          
    p_c = copy(x)          
    pi = copy(x)
    d = copy(x)          
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
    @debug "trust_region_" string(initial_x) (df isa OnceDifferentiable) (df isa AbstractObjective)
    value_jacobian!!(df, cache.x)
    @debug "trust_region_ evaluated at " string(cache.x) string(df.F) string(df.DF)
    
    cache.r .= NLsolve.value(df)

    NLsolve.check_isfinite(cache.r)

    @debug "trust_region_" string(cache.r) string(NLsolve.value(df)) string(ftol)
    it = 0
    x_converged, f_converged = assess_convergence(ustrip(initial_x), ustrip(cache.xold), ustrip(NLsolve.value(df)), NaN, ftol)

    stopped = any(isnan, cache.x) || any(isnan, NLsolve.value(df)) ? true : false

    converged = x_converged || f_converged
    delta = convert(real(T), NaN)
    rho = convert(real(T), NaN)
    if converged
        tr = NLsolve.SolverTrace()
        name = "Trust-region with dogleg"
        if autoscale
            name *= " and autoscaling"
        end
        return SolverResults(name, initial_x, reshape(cache.x, size(initial_x)...), norm(cache.r, Inf),
            initial_x, copy(cache.x), norm(cache.r, Inf),
            it, x_converged, xtol, f_converged, ftol, tr,
            first(df.f_calls), first(df.df_calls))
    end

    tr = NLsolve.SolverTrace()
    tracing = store_trace || show_trace || extended_trace
    NLsolve.@trustregiontrace convert(real(T), NaN)
    nn = length(cache.x)
    if autoscale
        for j = 1:nn
            jacrow = view(NLsolve.jacobian(df), :, j)
            rownorm = norm(jacrow./ oneunit.(jacrow))
            @debug "trust_region_ " string(jacrow) string(cache.d) string(rownorm)
            
            cache.d[j] = rownorm * oneunit(cache.d[j])
            if rownorm == zero(rownorm)
                cache.d[j] = oneunit(cache.d[j])
            end
        end
    else
        throw("hardly")
        # oneunit.(ArrayPartition(4.338005808987156kg, 1.4s))

        fill!(cache.d, one(real(T)))
    end
    @debug "trustregion_" string(cache.d) string(cache.x) 
    delta = factor * NLsolve.wnorm(cache.d ./ oneunit.(cache.d), cache.x ./ oneunit.(cache.x))
    @debug "trustregion_" it T delta
    if delta == zero(delta)
        delta = factor
    end

    eta = convert(real(T), 1e-4)

    while !stopped && !converged && it < iterations
        it += 1
       
        @debug "trustregion_" it T delta string(cache.p) string(cache.p_c) string(cache.r) string(cache.d)
        # Compute proposed iteration step
        dogleg!(cache.p, cache.p_c, cache.pi, cache.r, cache.d, NLsolve.jacobian(df), delta)
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
    @debug "trust_region LenNTRCache" string(xtol) string(ftol) iterations factor N
    trust_region_(df, initial_x, xtol, ftol, iterations, store_trace, show_trace, extended_trace, 
        convert(numtype(xtol), factor), autoscale, cache)
end
