using StructuredProximalOperators
using Random
using LinearAlgebra
using FiniteDifferences


function main()
    g = regularizer_lnuclear(1.0)
    Random.seed!(1234)
    x = rand(5, 6)
    Random.seed!(1456)
    eta = rand(5, 6)


    y, M = prox_αg(g, x, 0.6)
    @show M

    # Input curve, image by prox.
    function retract(M::FixedRankMatrices{m,n,k}, q) where {m,n,k}
        F = svd(q)
        return F.U[:, 1:k] * Diagonal(F.S[1:k]) * F.Vt[1:k, :]
    end
    c(t) = x + t * eta

    for t in 10 .^ range(-8, stop = 0, length = 20)
        y_t, M_t = prox_αg(g, c(t), 0.6)
        @show M_t
    end

    # DT(prox(x))[eta] by finite differences:
    f(x) = prox_αg(g, x, 0.6)[1]
    f_vec(x) = vec(f(x))
    DT_x = FiniteDifferences.jacobian(central_fdm(5, 1), f, x)[1]
    function DT_x_at(eta)
        eta_vec, vectomat = to_vec(eta)
        return vectomat(DT_x * eta_vec)
    end
    DT_x_at_eta = DT_x_at(eta)

    errorfinitediff(t) = norm(f(x + t * eta) - y - t * DT_x_at_eta)


    # DT(prox(x))[eta] by theoretical formula:




    comparison = compare_curves(errorfinitediff)
    return display_curvescomparison(comparison)
end


main()
