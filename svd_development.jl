using StructuredProximalOperators
using LinearAlgebra
using Random
using Manifolds

function get_Ṡ(::FixedRankMatrices{m,n,k,ℝ}, x, ξ) where {m,n,k}
    return Vector([ξ.M[i, i] for i in 1:k])
end

function get_U̇(::FixedRankMatrices{m,n,k,ℝ}, x, ξ) where {m,n,k}
    F = zeros(k, k)
    for i in 1:k, j in 1:k
        if x.S[i] != x.S[j]
            F[i, j] = 1 / (x.S[j]^2 - x.S[i]^2)
        end
    end

    return x.U * (F .* (ξ.M * Diagonal(x.S) + Diagonal(x.S) * ξ.M')) +
           ξ.U * Diagonal(x.S .^ (-1))
end
function get_V̇t(::FixedRankMatrices{m,n,k,ℝ}, x, ξ) where {m,n,k}
    F = zeros(k, k)
    for i in 1:k, j in 1:k
        if x.S[i] != x.S[j]
            F[i, j] = 1 / (x.S[j]^2 - x.S[i]^2)
        end
    end

    return (-F .* (Diagonal(x.S) * ξ.M + ξ.M' * Diagonal(x.S))) * x.Vt +
           Diagonal(x.S .^ (-1)) * ξ.Vt
end


function main()
    m, n = 5, 6
    k = 3
    M = FixedRankMatrices{m,n,k,ℝ}()

    Random.seed!(152)
    x = randomMPoint(M)
    Random.seed!(4158)
    ξ = randomTVector(M, x)

    x_emb = embed(M, x)
    ξ_emb = embed(M, x, ξ)


    γ(t) = retract(M, x, t * ξ)
    function errors_retraction(t)
        return norm(embed(M, γ(t)) - x_emb - t * ξ_emb)
    end
    function errors_proj_retraction(t)
        return norm(project(M, x, embed(M, γ(t)) - x_emb - t * ξ_emb))
    end

    #
    ## Symmetric term
    #
    S = x.S
    Ṡ = get_Ṡ(M, x, ξ)
    function errorsS(t)
        F = svd(embed(M, γ(t)))
        return norm(F.S[1:k] - (S + t * Ṡ))
    end

    #
    ## U term
    #
    U = x.U
    U̇ = get_U̇(M, x, ξ)
    function errorsU(t)
        F = svd(embed(M, γ(t)))
        return norm(F.U[:, 1:k] - (U + t * U̇))
    end

    #
    ## Vt term
    #
    Vt = x.Vt
    V̇t = get_V̇t(M, x, ξ)
    function errorsVt(t)
        F = svd(embed(M, γ(t)))
        return norm(F.Vt[1:k, :] - (Vt + t * V̇t))
    end

    comparison = compare_curves(
        errors_retraction,
        errors_proj_retraction,
        errorsU,
        errorsS,
        errorsVt,
    )
    return display_curvescomparison(comparison)
end

main()
