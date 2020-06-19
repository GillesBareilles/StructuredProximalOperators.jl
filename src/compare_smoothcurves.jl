

function compare_curves(args...; tmin = 1e-8, tmax = 1, npoints = 50)
    ts = 10 .^ range(log(10, tmin), stop = log(10, tmax), length = npoints)

    curves = Dict(Symbol(r) => [] for r in args if r isa Function)
    curves[:slope2] = []
    curves[:slope3] = []

    for (i, t) in enumerate(ts)
        for curve in args
            if curve isa Function
                val = curve(t)
                @assert val >= 0 "curve should be positive, value is $(curve(t))."
                push!(curves[Symbol(curve)], (t, val))
            end
        end
        push!(curves[:slope2], (t, t^2))
        push!(curves[:slope3], (t, t^3))
    end
    return curves
end

function display_curvescomparison(curves)
    ps = []

    for (k, v) in curves
        legendentry = string(k)
        opts = occursin("slope", legendentry) ? Dict("no marks" => nothing) : Dict()
        push!(ps, PlotInc(PGFPlotsX.Options(opts...), Coordinates(v)))
        push!(ps, LegendEntry(replace(string(k), "_" => " ")))
    end


    ## Plot errors:
    return fig_time_subopt = @pgf Axis(
        {
            xmode = "log",
            ymode = "log",
            height = "12cm",
            width = "12cm",
            xlabel = "t",
            # ylabel = L"f(R_x(t\eta))-f(x)-t\langle gradf(x),\eta\rangle",
            legend_pos = "outer north east",
        },
        ps...,
    )
end


function check_regularizer_gradient_hessian(M, reg)
    x = randomMPoint(M)
    p = embed(M, retract(M, x, randomTVector(M, x)))
    ξ = randomTVector(M, x)
    η = randomTVector(M, x)

    f(x) = reg(x)
    gradf_x = ∇M_g(reg, M, x)
    hessf_x_ξ = ∇²M_g_ξ(reg, M, x, ξ)
    hessf_x_η = ∇²M_g_ξ(reg, M, x, η)

    # 1. Gradient is tangent.
    println("- gradf(x) ∈ T_x M:\t\t\t\t", is_tangent_vector(M, x, gradf_x; atol = 1e-10))

    # 2. Hessian is symetric
    println(
        "- Hess f(x)[η] ∈ T_x M:\t\t\t\t",
        is_tangent_vector(M, x, hessf_x_η, atol = 1e-10),
    )
    println(
        "- Hess f(x)[ξ] ∈ T_x M:\t\t\t\t",
        is_tangent_vector(M, x, hessf_x_ξ, atol = 1e-10),
    )
    println(
        "- ⟨Hess f(x)[ξ], η⟩ - ⟨Hess f(x)[η], ξ⟩:\t",
        inner(M, x, hessf_x_η, ξ) - inner(M, x, hessf_x_ξ, η),
    )

    f_x = f(embed(M, x))
    ηgradfx = inner(M, x, η, gradf_x)
    ηHessf_xη = inner(M, x, η, hessf_x_η)

    @show ηgradfx, ηHessf_xη

    frgrad(t) = abs(f(embed(M, retract(M, x, t * η))) - (f(embed(M, x)) + t * ηgradfx))
    function frhess(t)
        return abs(
            f(embed(M, retract(M, x, t * η))) - (f_x + t * ηgradfx + 0.5 * t^2 * ηHessf_xη),
        )
    end

    comparison = compare_curves(frgrad, frhess)
    return display_curvescomparison(comparison)
end


function check_e2r_gradient_hessian(M)
    x = randomMPoint(M)
    p = embed(M, retract(M, x, randomTVector(M, x)))

    ## function, rgrad, rhess
    f(x) = exp(sum(x .* (x .- p) .^ 2))

    function get_rgrad(M::Manifold, x)
        egrad = ForwardDiff.gradient(f, embed(M, x))
        return egrad_to_rgrad(M, x, egrad)
    end

    function get_rhess(M::Manifold, x, η)
        x_emb = embed(M, x)
        η_emb = embed(M, x, η)
        ∇f_x = ForwardDiff.gradient(f, x_emb)
        ∇²f_x_η =
            reshape(ForwardDiff.hessian(f, x_emb) * vec(η_emb), representation_size(M))
        return ehess_to_rhess(M, x, ∇f_x, ∇²f_x_η, η)
    end

    ## Gradient check
    η = randomTVector(M, x)

    # f1(t) =
    # f_x = f(x)
    gradf_x = get_rgrad(M, x)

    # 1. gradf ∈ T_x M
    println("- gradf(x) ∈ T_x M:\t\t\t\t", is_tangent_vector(M, x, gradf_x; atol = 1e-10))
    println("- η ∈ T_x M:\t\t\t\t\t", is_tangent_vector(M, x, η; atol = 1e-10))

    # 2. Hessian is symetric
    η = randomTVector(M, x)
    ξ = randomTVector(M, x)

    hessf_x_η = get_rhess(M, x, η)
    hessf_x_ξ = get_rhess(M, x, ξ)

    println(
        "- Hess f(x)[η] ∈ T_x M:\t\t\t\t",
        is_tangent_vector(M, x, hessf_x_η, atol = 1e-10),
    )
    println(
        "- Hess f(x)[ξ] ∈ T_x M:\t\t\t\t",
        is_tangent_vector(M, x, hessf_x_ξ, atol = 1e-10),
    )
    println(
        "- ⟨Hess f(x)[ξ], η⟩ - ⟨Hess f(x)[η], ξ⟩:\t",
        inner(M, x, hessf_x_η, ξ) - inner(M, x, hessf_x_ξ, η),
    )

    f_x = f(embed(M, x))
    ηgradfx = inner(M, x, η, gradf_x)
    ηHessf_xη = inner(M, x, get_rhess(M, x, η), η)

    @show ηgradfx, ηHessf_xη

    frgrad(t) = abs(f(embed(M, retract(M, x, t * η))) - (f(embed(M, x)) + t * ηgradfx))
    function frhess(t)
        return abs(
            f(embed(M, retract(M, x, t * η))) - (f_x + t * ηgradfx + 0.5 * t^2 * ηHessf_xη),
        )
    end

    comparison = compare_curves(frgrad, frhess)
    return display_curvescomparison(comparison)
end


function check_retraction(M)
    Random.seed!(1234)
    x = randomMPoint(M)
    Random.seed!(4312)
    η = randomTVector(M, x)

    x_emb = embed(M, x)
    η_emb = embed(M, x, η)

    function retractiontη(t)
        retracted = retract(M, x, t * η)
        return embed(M, retracted)
    end
    xtη(t) = x_emb + t * η_emb

    function retractionerror(t; x = x)
        return norm(project(M, x, retractiontη(t) - xtη(t)))
    end

    comparison = compare_curves(retractionerror)
    return display_curvescomparison(comparison)
end
