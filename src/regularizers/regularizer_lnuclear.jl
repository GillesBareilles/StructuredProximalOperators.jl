
##
## Nuclear norm regularization
##
@with_kw mutable struct regularizer_lnuclear <: Regularizer
    λ::Float64
end

function wholespace_manifold(reg::regularizer_lnuclear, x)
    m, n = size(x)
    return FixedRankMatrices(m, n, min(m, n))
end

## 0th order
function g(reg::regularizer_lnuclear, x)
    return reg.λ * norm(svd(x).S, 1)
end

function g(reg::regularizer_lnuclear, x::SVDMPoint)
    return reg.λ * sum(x.S)
end

## 1st order
function prox_αg!(g::regularizer_lnuclear, res::AbstractMatrix, x, α)
    F = svd(x)
    st_spectrum = softthresh.(F.S, g.λ * α)
    k = count(x -> x > 0, st_spectrum)
    m, n = size(x)
    res .= F.U * Diagonal(st_spectrum) * F.Vt

    return FixedRankMatrices(m, n, k, ℝ)
end

function prox_αg!(g::regularizer_lnuclear, res::SVDMPoint, x, α)
    F = svd(x)
    m, n = size(x)
    res.U .= F.U
    res.Vt .= F.Vt
    res.S .= softthresh.(F.S, g.λ * α)

    k = count(x -> x > 0, res.S)
    return FixedRankMatrices(m, n, k, ℝ)
end

# function prox_αg(g::regularizer_lnuclear, x, α)
#     F = svd(x)
#     st_spectrum = softthresh.(F.S, g.λ * α)
#     k = count(x -> x > 0, st_spectrum)
#     m, n = size(x)

#     return F.U * Diagonal(st_spectrum) * F.Vt, FixedRankMatrices(m, n, k, ℝ)
# end

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
function ∇M_g!(
    g::regularizer_lnuclear,
    ::FixedRankMatrices{m,n,k},
    grad_g::AbstractArray,
    x::AbstractArray,
) where {m,n,k}
    F = svd(x)

    grad_g .= 0
    for i in 1:k
        grad_g .+= g.λ * F.U[i, :] * F.Vt[i, :]'
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
function ∇M_g(g::regularizer_lnuclear, M::FixedRankMatrices{m,n,k}, x::AbstractArray) where {m,n,k}
    grad_g = zeros(representation_size(M))
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

function ∇²M_g_ξ!(
    g::regularizer_lnuclear,
    M::FixedRankMatrices{m,n,k},
    hess_gxξ,
    x::Array{Float64,2},
    ξ::Array{Float64,2},
) where {m,n,k}

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




#
### Optimality status
#
function model_g_subgradient!(model, regularizer::regularizer_lnuclear, M::FixedRankMatrices{m, n, k}, x) where {m, n, k}
    λ = regularizer.λ

    @variable(model, ḡ_normal[1:min(m, n)-k])
    @constraint(model, 0 .<= ḡ_normal .<= λ)

    return ḡ_normal
end

function build_subgradient_from_normalcomp(regularizer::regularizer_lnuclear, M::FixedRankMatrices{m, n, k}, x, ḡ_normal) where {m, n, k}
    ḡ = Matrix(undef, m, n)
    λ = regularizer.λ

    if length(ḡ_normal) == 0
        ḡ .= 0.0
    else
        fill_ḡ!(ḡ, x)
    end
    return ḡ
end

function fill_ḡ!(ḡ, x)
    F = svd(x)
    S = vcat(ones(k), ḡ_normal)
    for i in 1:m, j in 1:n
        ḡ[i, j] = dot(F.U[i, :] .* S, F.Vt[:, j])
    end
    return
end

function fill_ḡ!(ḡ, x::SVDMPoint)
    S = vcat(ones(k), ḡ_normal)
    for i in 1:m, j in 1:n
        ḡ[i, j] = dot(x.U[i, :] .* S, x.Vt[:, j])
    end
end
