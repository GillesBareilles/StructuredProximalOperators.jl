using StructuredProximalOperators
using LinearAlgebra
using Random

function get_U̇(M, m, n, k, x, ξ)
    F = zeros(k, k)
    for i in 1:k, j in 1:k
        if x.S[i] != x.S[j]
            F[i, j] = 1 / (x.S[j] - x.S[i])
        end
    end

    # return x.U * (F .* (ξ.M*Diagonal(x.S) + Diagonal(x.S)*ξ.M')) + ξ.U*Diagonal(x.S.^(-1))

    ξ_emb = embed(M, x, ξ)
    return x.U * (
        F .* (x.U' * ξ_emb * x.Vt' * Diagonal(x.S) + Diagonal(x.S) * x.Vt * ξ_emb' * x.U)
    ) + (Diagonal(ones(m)) - x.U * x.U') * ξ_emb * x.Vt' * inv(Diagonal(x.S))
end


function main()
    m, n = 5, 6
    k = 3
    M = FixedRankMatrices{m,n,k,ℝ}()
    Random.seed!(152)
    x = randomMPoint(M)
    Random.seed!(4158)
    ξ = randomTVector(M, x)


    ###
    # Handling svd complications
    x_emb = embed(M, x)
    F_x = svd(x_emb)

    x = SVDMPoint(F_x.U[:, 1:k], F_x.S[1:k], F_x.Vt[1:k, :])

    ξ_emb = embed(M, x, ξ)
    ###

    γ(t) = retract(M, x, t * ξ)

    U = x.U
    Vt = x.Vt


    U̇ = get_U̇(M, m, n, k, x, ξ)

    function errorsU(t)
        F = svd(x_emb + t * ξ_emb)
        return norm(F.U[:, 1:k] - (U + t * U̇))
    end

    comparison = compare_curves(errorsU)
    return display_curvescomparison(comparison)
end

main()
