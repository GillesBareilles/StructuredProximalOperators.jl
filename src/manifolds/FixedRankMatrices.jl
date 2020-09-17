
function <(
    ::FixedRankMatrices{m1,n1,k1,ℝ},
    ::FixedRankMatrices{m2,n2,k2,ℝ},
) where {m1,n1,k1,m2,n2,k2}
    return m1 == m2 && n1 == n2 && k1 < k2
end

copy(::FixedRankMatrices{m,n,k,ℝ}) where {m,n,k} = FixedRankMatrices(m, n, k)

embedding_dimension(::FixedRankMatrices{m,n,k,ℝ}) where {m,n,k} = m*n

function randomMPoint(M::FixedRankMatrices{m,n,k,ℝ}) where {m,n,k}
    A = rand(m, n)
    F = svd(A)
    # We need this trick to ensure the point is 'qualified' and continuity of svd.
    A .= F.U[:, 1:k] * Diagonal(F.S[1:k] .+ 0.5) * F.Vt[1:k, :]
    return SVDMPoint(A, k)
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
    ::FixedRankMatrices{m,n,k,ℝ},
    ξ::AbstractMatrix,
    p::SVDMPoint,
    A,
) where {m,n,k}
    av = A * (p.Vt')
    uTav = p.U' * av
    aTu = A' * p.U
    ξ .= p.U * uTav * p.Vt + (A * p.Vt' - p.U * uTav) * p.Vt + p.U * (aTu - p.Vt' * uTav')'
    return ξ
end

function project!(
    M::FixedRankMatrices{m,n,k,ℝ},
    ξ::AbstractMatrix,
    x::AbstractMatrix,
    A,
) where {m,n,k}
    F = svd(x)

    innerdecomp = F.U' * A * F.Vt'
    innerdecomp[(k + 1):end, (k + 1):end] .= 0
    ξ = F.U * innerdecomp * F.Vt
    return ξ
end

function project!(
    M::FixedRankMatrices{m, n, k, ℝ},
    res::Array{Float64},
    A::Array{Float64},
) where {m,n,k}
    F = svd(A)
    truncatedsingvals = F.S
    F.S[k+1:end] .= 0
    res .= F.U * Diagonal(truncatedsingvals) * F.Vt
    return res
end

# function zero_tangent_vector(M::FixedRankMatrices{m,n,k,ℝ}, x::AbstractArray) where {m,n,k}
#     return zeros(representation_size(M))
# end
