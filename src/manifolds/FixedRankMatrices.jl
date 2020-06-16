
function randomMPoint(M::FixedRankMatrices{m,n,k,ℝ}) where {m,n,k}
    A = rand(m, n)
    F = svd(A)
    return SVDMPoint(F.U[:, 1:k], F.S[1:k] .+ 0.5, F.Vt[1:k, :])
end

function randomTVector(M::FixedRankMatrices{m,n,k,ℝ}, x) where {m,n,k}
    ξ = zero_tangent_vector(M, x)
    project!(M, ξ, x, rand(m, n))
    return ξ /= norm(M, x, ξ)
end

embed!(::FixedRankMatrices, q, p) = (q = p.U * Diagonal(p.S) * p.Vt)
function embed(M::FixedRankMatrices{m,n,k,ℝ}, p::SVDMPoint) where {m,n,k}
    p_embed = zeros(m, n)
    return embed!(M, p_embed, p)
end

function embed!(::FixedRankMatrices, ξ_embed, p, ξ)
    return (ξ_embed = p.U * ξ.M * p.Vt + ξ.U * p.Vt + p.U * ξ.Vt)
end
function embed(M::FixedRankMatrices{m,n,k,ℝ}, p::SVDMPoint, ξ::UMVTVector) where {m,n,k}
    ξ_embed = zeros(m, n)
    return embed!(M, ξ_embed, p, ξ)
end

function ehess_to_rhess(
    ::FixedRankMatrices{m,n,k,ℝ},
    x::SVDMPoint,
    ∇f_x,
    ∇²f_ξ,
    ξ::UMVTVector,
) where {m,n,k}
    res = UMVTVector(
        (Diagonal(ones(m)) - x.U * x.U') *
        (∇²f_ξ * x.Vt' + ∇f_x * ξ.Vt' * Diagonal(x.S .^ -1)),
        x.U' * ∇²f_ξ * x.Vt',
        (
            (Diagonal(ones(n)) - x.Vt' * x.Vt) *
            (∇²f_ξ' * x.U + ∇f_x' * ξ.U * Diagonal(x.S .^ -1))
        )',
    )
    return res
end


function inner(
    ::FixedRankMatrices{m,n,k,ℝ},
    x,
    ξ::AbstractMatrix,
    η::AbstractMatrix,
) where {m,n,k}
    return dot(ξ, η)
end

# retract(M::FixedRankMatrices{m,n,k,ℝ}, x, ξ) where {m,n,k} = retract(M, x, ξ, PolarRetraction())
function retract(M::FixedRankMatrices{m,n,k,ℝ}, x, ξ) where {m,n,k}
    F = svd(embed(M, x) + embed(M, x, ξ))
    return SVDMPoint(F.U[:, 1:k], F.S[1:k], F.Vt[1:k, :])
end

function project!(
    M::FixedRankMatrices{m,n,k,ℝ},
    ξ::AbstractMatrix,
    p,
    A::AbstractMatrix,
) where {m,n,k}
    av = A * (p.Vt')
    uTav = p.U' * av
    aTu = A' * p.U
    ξ .=
        p.U * uTav * p.Vt + (A * p.Vt' - p.U * uTav) * p.Vt + p.U * ((aTu - p.Vt' * uTav')')
    return ξ
end
