using StructuredProximalOperators
using Random
using LinearAlgebra
using FiniteDifferences
using ForwardDiff

function optimize_proxgrad(pb, x)


function main()
    m, n = 5, 6
    k = 4
    α = 2.0
    g = regularizer_lnuclear(1.0)


    M = FixedRankMatrices(m, n, k, ℝ)
    Random.seed!(1234)
    x = randomMPoint(M)
    Random.seed!(1456)
    η = randomTVector(M, x)
    x_emb = embed(M, x)
    η_emb = embed(M, x, η)


    y, N = prox_αg(g, x, α)
    y_emb = embed(M, y)
    @show M
    @show N


    # Input curve, image by prox.
    c(t) = retract(M, x, t * η)

    ## Check the prox image lies in the same manifold for considered t range
    # for t in 10 .^ range(-8, stop = 0, length = 20)
    #     y_t, M_t = prox_αg(g, c(t), α)
    #     @show M_t
    # end

    # 1. Approximate DT(prox(x))[η] by finite differences:
    T(x) = prox_αg(g, x, α)[1]
    DT_x = FiniteDifferences.jacobian(central_fdm(5, 1), T, x_emb)[1]
    function DT_x_at(η)
        η_vec, vectomat = to_vec(η)
        return vectomat(DT_x * η_vec)
    end

    DT_x_η_emb = DT_x_at(η_emb)
    DT_x_η = zero_tangent_vector(N, y)
    project!(N, DT_x_η, y, DT_x_η_emb)
    println("dist(DT(x)[η],  T_y M):\t\t", norm(DT_x_η_emb - embed(N, y, DT_x_η)))

    finitediff_prox(t) = norm(T(x_emb + t * η_emb) - y_emb - t * DT_x_η_emb)


    # 2. Approximate projection derivative by finite differences:
    function proj(x, v)
        return project(M_opt, x, v)
    end
    proj_v(x) = proj(x, y_emb - x_emb)

    println("")
    println("")

    Dproj_Tc = FiniteDifferences.jacobian(central_fdm(4, 1), t -> proj_v(T(c(t))), 0.0)[1]
    _, vectomat = to_vec(rand(5, 6))
    Dproj_Tc = vectomat(Dproj_Tc)

    finitediff_proj(t) = norm(proj_v(T(c(t))) - proj_v(T(c(0))) - t * Dproj_Tc)

    # @show proj(y, y_emb-x_emb)



    # 2. Check DT(prox(x))[η] theoretical formula:
    Hess_DT = zero_tangent_vector(N, y)
    ∇²M_g_ξ!(g, N, Hess_DT, y, DT_x_η)
    Hess_DT_emb = embed(M, y, Hess_DT)

    η_proj = project(N, y, η_emb)
    η_proj_emb = project(N, y, η_proj)
    @show norm(DT_x_η_emb + α * Hess_DT_emb + Dproj_Tc - project(N, y, η_emb))
    @show norm(DT_x_η_emb + α * Hess_DT_emb + Dproj_Tc)


    comparison = compare_curves(finitediff_prox, finitediff_proj)
    return display_curvescomparison(comparison)
end


main()
