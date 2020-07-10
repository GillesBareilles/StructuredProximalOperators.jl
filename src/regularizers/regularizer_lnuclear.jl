
##
## Nuclear norm regularization
##
@with_kw mutable struct regularizer_lnuclear <: Regularizer
    λ::Float64
end

function wholespace_manifold(reg::regularizer_lnuclear, x)
    m, n = size(x)
    return FixedRankMatrices(m, n, max(m, n))
end

## 0th order
function g(reg::regularizer_lnuclear, x)
    return reg.λ * norm(svd(x).S, 1)
end

function g(reg::regularizer_lnuclear, x::SVDMPoint)
    return reg.λ * sum(x.S)
end

## 1st order
function prox_αg(g::regularizer_lnuclear, x, α)
    F = svd(x)
    st_spectrum = softthresh.(F.S, g.λ * α)
    k = count(x -> x > 0, st_spectrum)
    m, n = size(x)

    return F.U * Diagonal(st_spectrum) * F.Vt, FixedRankMatrices(m, n, k, ℝ)
end

function prox_αg(g::regularizer_lnuclear, x::SVDMPoint, α)
    st_spectrum = softthresh.(x.S, g.λ * α)
    k = count(x -> x > 0, st_spectrum)
    m = size(x.U, 1)
    n = size(x.Vt, 2)
    return SVDMPoint(x.U[:, 1:k], st_spectrum[1:k], x.Vt[1:k, :]),
    FixedRankMatrices(m, n, k, ℝ)
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
function ∇²M_g_ξ!(
    g::regularizer_lnuclear,
    M::FixedRankMatrices{m,n,k},
    hess_gxξ,
    x::SVDMPoint,
    ξ::UMVTVector,
) where {m,n,k}
    F = zeros(k, k)
    @inbounds for i in 1:k, j in 1:k
        if x.S[i] != x.S[j]
            F[i, j] = 1 / (x.S[j]^2 - x.S[i]^2)
        end
    end

    ## 1. working implementation; litteral formula
    U̇ =
        x.U * (F .* (ξ.M * Diagonal(x.S) + Diagonal(x.S) * ξ.M')) +
        ξ.U * Diagonal(x.S .^ (-1))
    V̇t =
        (-F .* (Diagonal(x.S) * ξ.M + ξ.M' * Diagonal(x.S))) * x.Vt +
        Diagonal(x.S .^ (-1)) * ξ.Vt

    project!(M, hess_gxξ, x, g.λ * (U̇ * x.Vt + x.U * V̇t))

    # TODO: optimize this.

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
