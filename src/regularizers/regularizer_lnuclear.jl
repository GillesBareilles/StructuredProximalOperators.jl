
##
## Nuclear norm regularization
##
struct regularizer_lnuclear <: Regularizer
    λ::Float64
end


## 0th order
function (g::regularizer_lnuclear)(x)
    return g.λ * norm(svd(x).S, 1)
end

function (g::regularizer_lnuclear)(x::SVDMPoint)
    return sum(x.S)
end

## 1st order
function prox_αg(g::regularizer_lnuclear, x, α)
    F = svd(x)
    st_spectrum = softthresh.(F.S, g.λ * α)
    k = count(x -> x > 0, st_spectrum)
    m, n = size(x)

    return F.U * Diagonal(st_spectrum) * F.Vt, FixedRankMatrices(m, n, k, ℝ)
end

function ∇M_g!(
    g::regularizer_lnuclear,
    ::FixedRankMatrices{m,n,k},
    grad_g::UMVTVector,
    x,
) where {m,n,k}
    grad_g.M .= 0
    for i in 1:k
        grad_g.M[i, i] = g.λ
    end
    return grad_g
end
function ∇M_g(
    g::regularizer_lnuclear,
    M::FixedRankMatrices{m,n,k},
    x::SVDMPoint,
) where {m,n,k}
    grad_g = zero_tangent_vector(M, x)
    return ∇M_g!(g, M, grad_g, x)
end
function ∇M_g(g::regularizer_lnuclear, M::FixedRankMatrices{m,n,k}, x) where {m,n,k}
    grad_g = zero_tangent_vector(M, randomMPoint(M))            # ! dirty fix, to clean up with proper introduction of containers
    return ∇M_g!(g, M, grad_g, x)
end


## 2nd order
# function ∇²M_g_ξ!(g::regularizer_lnuclear, M::FixedRankMatrices{m,n,k}, hess_gxξ, x, ξ) where {m,n,k}
#     #! NOTE: This should be done with Gram Schmidth, without forming full matrix...
#     Ufull = zeros(m, m)
#     Ufull[:, 1:k] .= x.U
#     Uperp = nullspace(Ufull')

#     Vfull = zeros(n, n)
#     Vfull[1:k, :] .= x.Vt
#     tVperp = nullspace(Vfull)'

#     B₁ = Uperp' * embed(M, x, ξ) * x.Vt' * Diagonal(x.S.^(-1))
#     tB₂ = Diagonal(x.S.^(-1)) * x.U' * embed(M, x, ξ) * tVperp'

#     hess_gxξ.M .= 0
#     hess_gxξ.U .= Uperp * B₁
#     hess_gxξ.Vt .= tB₂ * tVperp
#     return hess_gxξ
# end
function ∇²M_g_ξ!(
    g::regularizer_lnuclear,
    M::FixedRankMatrices{m,n,k},
    hess_gxξ,
    x::SVDMPoint,
    ξ,
) where {m,n,k}
    hess_gxξ.M .= 0

    F = zeros(k, k)
    for i in 1:k, j in 1:k
        if x.S[i] != x.S[j]
            F[i, j] = 1 / (x.S[j] - x.S[i])
        end
    end

    hess_gxξ.U .=
        x.U * (F .* (ξ.M * Diagonal(x.S) + Diagonal(x.S) * ξ.M')) +
        ξ.U * Diagonal(x.S .^ -1)
    hess_gxξ.Vt .=
        (
            x.Vt' * (F .* (Diagonal(x.S) * ξ.M + ξ.M' * Diagonal(x.S))) +
            ξ.Vt' * Diagonal(x.S .^ -1)
        )'


    ## old version
    F = svd(embed(M, x), full = true)

    U = F.U[:, 1:k]
    Uperp = F.U[:, (k + 1):end]
    tV = F.Vt[1:k, :]
    tVperp = F.Vt[(k + 1):end, :]
    Σ = Diagonal(F.S[1:k])

    B₁ = transpose(Uperp) * embed(M, x, ξ) * transpose(tV) * inv(Σ)
    tB₂ = inv(Σ) * transpose(U) * embed(M, x, ξ) * transpose(tVperp)

    res_embed = Uperp * B₁ * tV + U * tB₂ * tVperp
    hess_gxξ2 = zero_tangent_vector(M, x)
    project!(M, hess_gxξ2, x, res_embed)

    println("------")
    display(hess_gxξ)
    println("  ---   ")
    display(hess_gxξ2)
    # return hess_gxξ2

    @error false "not correctly implemented yet."

    return hess_gxξ
end

function ∇²M_g_ξ(
    g::regularizer_lnuclear,
    M::FixedRankMatrices{m,n,k},
    x::SVDMPoint,
    ξ,
) where {m,n,k}
    hess_gxξ = zero_tangent_vector(M, x)
    return ∇²M_g_ξ!(g, M, hess_gxξ, x, ξ)
end
function ∇²M_g_ξ(g::regularizer_lnuclear, M::FixedRankMatrices{m,n,k}, x, ξ) where {m,n,k}
    hess_gxξ = zero_tangent_vector(M, randomMPoint(M))          # !dirty fix, to clean with proper storage diff of points / vectors.
    return ∇²M_g_ξ!(g, M, hess_gxξ, x, ξ)
end
