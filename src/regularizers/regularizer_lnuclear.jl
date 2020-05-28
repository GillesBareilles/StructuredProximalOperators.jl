
##
## Nuclear norm regularization
##
struct regularizer_lnuclear <: AbstractRegularizer end


## 0th order
function (::regularizer_lnuclear)(x)
    return norm(svd(x).S, 1)
end

## 1st order
function prox_αg(g::regularizer_lnuclear, x, α)
    F = svd(x)
    st_spectrum = softthresh.(F.S, α)
    k = count(x -> x > 0, st_spectrum)
    m, n = size(x)

    return F.U * Diagonal(softthresh.(F.S, α)) * F.Vt, FixedRankMatrices(m, n, k)
end

function ∇M_g(g::regularizer_lnuclear, M::FixedRankMatrices{m,n,k,𝔽}, x) where {m,n,k,𝔽}
    F = svd(x)
    return F.U[:, 1:k] * F.Vt[1:k, :]
end


## 2nd order
function ∇²M_g_ξ(
    g::regularizer_lnuclear,
    M::FixedRankMatrices{m,n,k,𝔽},
    x,
    ξ,
) where {m,n,k,𝔽}
    F = svd(x, full = true)

    U = F.U[:, 1:k]
    Uperp = F.U[:, k+1:end]
    tV = F.Vt[1:k, :]
    tVperp = F.Vt[k+1:end, :]
    Σ = Diagonal(F.S[1:k])

    B₁ = transpose(Uperp) * ξ * transpose(tV) * inv(Σ)
    tB₂ = inv(Σ) * transpose(U) * ξ * transpose(tVperp)

    return Uperp * B₁ * tV + U * tB₂ * tVperp
end
