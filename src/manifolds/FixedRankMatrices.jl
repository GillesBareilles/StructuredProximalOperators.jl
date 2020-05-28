# import Base: exp, log, show, copy
# import LinearAlgebra: norm, dot

# using Base: eps
# using LinearAlgebra: svd, Diagonal, rank, diag, diagm, eigen, eigvals, eigvecs, tr, triu, qr, cholesky, Hermitian


# Types
# ---
struct FixedRank{m,n,k} <: AbstractManifold where {m,n,k}
    name::String
    abbreviation::String
    FixedRank{m,n,k}() where {m,n,k} = new("Fixed $k-rank of $m×$n matrices", "$m×$n - $k")
end
FixedRank(m::Int, n::Int, k::Int) = FixedRank{m,n,k}()


const MatrixFixedRankMPoint = Matrix
const MatrixFixedRankTVector = Matrix

const VectorFixedRankMPoint = Vector
const VectorFixedRankTVector = Vector

copy(M::FixedRank{m,n,k}) where {m,n,k} = FixedRank{m,n,k}()


# Functions
# ---
manifold_dimension(M::FixedRank{m,n,k}) where {m,n,k} = k(m + n - k)

function randomMPoint(M::FixedRank{m,n,k}) where {m,n,k}
    A = rand(m, n)
    F = svd(A)
    singvals = F.S
    singvals[k+1:end] .= 0
    return F.U * Diagonal(singvals) * F.Vt
end

function randomTVector(M::FixedRank{m,n,k}, x::MatrixFixedRankMPoint) where {m,n,k}
    F = svd(x)

    M = rand(size(F.U, 2), size(F.Vt, 1))
    M[k+1:end, k+1:end] .= 0

    res = F.U * M * F.Vt
    res /= norm(res)
    return res
end


# metric
inner(M::FixedRank, x, ξ, ν) = dot(ξ, ν)
norm(M::FixedRank, x, ξ) = inner(M, x, ξ, ξ)

# Tangent space
function project!(
    M::FixedRank{m,n,k},
    ξ::MatrixFixedRankTVector,
    x::MatrixFixedRankMPoint,
    q::AbstractMatrix,
) where {m,n,k}
    F = svd(x)

    innerdecomp = transpose(F.U) * q * transpose(F.Vt)
    innerdecomp[k+1:end, k+1:end] .= 0
    ξ = F.U * innerdecomp * F.Vt
    return ξ
end
function project(
    M::FixedRank{m,n,k},
    x::MatrixFixedRankMPoint,
    q::AbstractMatrix,
) where {m,n,k}
    ξ = zeros(size(x))
    return project!(M, ξ, x, q)
end

## TODO: improve this implementation and the handling of different representations of points/vectors
function project(
    M::FixedRank{m,n,k},
    x::VectorFixedRankMPoint,
    q::AbstractVector,
) where {m,n,k}
    X = reshape(x, (m, n))
    Q = reshape(q, (m, n))
    return vec(project(M, X, Q))
end


function retract!(
    M::FixedRank{m,n,k},
    y::MatrixFixedRankMPoint,
    x::MatrixFixedRankMPoint,
    ξ::MatrixFixedRankTVector,
) where {m,n,k}
    F = svd(x + ξ)
    y = F.U[:, 1:k] * Diagonal(F.S[1:k]) * F.Vt[1:k, :]
    return y
end
function retract(
    M::FixedRank{m,n,k},
    x::MatrixFixedRankMPoint,
    ξ::MatrixFixedRankTVector,
) where {m,n,k}
    y = zeros(size(x))
    return retract!(M, y, x, ξ)
end


function retract(
    M::FixedRank{m,n,k},
    x::VectorFixedRankMPoint,
    ξ::VectorFixedRankTVector,
) where {m,n,k}
    X = reshape(x, (m, n))
    Ξ = reshape(ξ, (m, n))
    return vec(retract(M, X, Ξ))
end


# Euclidean to riemannian gradients, hessians at vectors.
egrad_to_rgrad!(M::FixedRank{m,n,k}, gradf_x, x, ∇f_x) where {m,n,k} =
    project!(M, gradf_x, x, ∇f_x)
egrad_to_rgrad(M::FixedRank{m,n,k}, x, ∇f_x) where {m,n,k} = project(M, x, ∇f_x)


function ehess_to_rhess!(
    M::FixedRank{m,n,k},
    Hessf_xξ,
    x::MatrixFixedRankMPoint,
    ∇f_x,
    ∇²f_ξ,
    ξ::MatrixFixedRankTVector,
) where {m,n,k}
    F = svd(x, full = true)

    U = F.U[:, 1:k]
    Uperp = F.U[:, k+1:end]
    tV = F.Vt[1:k, :]
    tVperp = F.Vt[k+1:end, :]
    Σ = Diagonal(F.S[1:k])

    B₁ = transpose(Uperp) * ξ * transpose(tV) * inv(Σ)
    tB₂ = inv(Σ) * transpose(U) * ξ * transpose(tVperp)

    tUperpξVperp = transpose(Uperp) * ξ * transpose(tVperp)

    project!(M, Hessf_xξ, x, x + ∇²f_ξ)
    Hessf_xξ += U * transpose(B₁) * tUperpξVperp * tVperp
    Hessf_xξ += Uperp * tUperpξVperp * transpose(tB₂) * tV
    return Hessf_xξ
end
function ehess_to_rhess(M::FixedRank, x, ∇f_x, ∇²f_ξ, ξ)                                    # ! this should factor out
    Hessf_xξ = zeros(size(ξ))
    return ehess_to_rhess!(M, Hessf_xξ, x, ∇f_x, ∇²f_ξ, ξ)
end


function ehess2rhess(
    M::FixedRank{m,n,k},
    x::VectorFixedRankMPoint,
    ∇f_x,
    ∇²f_ξ,
    ξ::VectorFixedRankTVector,
) where {m,n,k}
    X = reshape(x, (m, n))
    Ξ = reshape(ξ, (m, n))
    ∇f_x_mat = reshape(∇f_x, (m, n))
    ∇²f_ξ_mat = reshape(∇²f_ξ, (m, n))
    return vec(ehess_to_rhess(M, X, ∇f_x_mat, ∇²f_ξ_mat, Ξ))
end

# Validation
# ---
function check_manifold_point(M::FixedRank{m,n,k}, x::MatrixFixedRankMPoint) where {m,n,k}
    if size(x, 1) != m || size(x, 2) != n
        return DomainError(
            size(x),
            "The dimension of x must be ($m, $n) but it is $(size(x))",
        )
    end

    x_rank = rank(x)
    if x_rank != k
        return DomainError(x_rank, "x should have rank $k, but it is $x_rank.")
    end
    return nothing
end

## ! this should probably handle the dims of x as a vector...
function check_manifold_point(M::FixedRank{m,n,k}, x::VectorFixedRankMPoint) where {m,n,k}
    X = reshape(x, (m, n))
    return check_manifold_point(M, X)
end

function check_tangent_vector(
    M::FixedRank{m,n,k},
    x::MatrixFixedRankMPoint,
    ξ::MatrixFixedRankTVector,
) where {m,n,k}
    check_manifold_point(M, x)

    if size(ξ) != (m, n)
        return DomainError(
            size(ξ),
            "The dimension of ξ must be ($m, $n) but it is $(size(x))",
        )
    end

    F = svd(x)
    innerdecomp = transpose(F.U) * ξ * transpose(F.Vt)
    orthcomp = norm(innerdecomp[k+1:end, k+1:end])
    if orthcomp > 5e-13
        return DomainError(
            norm(innerdecomp[k+1:end, k+1:end]),
            "ξ has nonzero norm ($orthcomp) component on the orthogonal space.",
        )
    end
    return nothing
end

## ! this should probably handle x and ξ as vectors...
function check_tangent_vector(
    M::FixedRank{m,n,k},
    x::VectorFixedRankMPoint,
    ξ::VectorFixedRankTVector,
) where {m,n,k}
    X = reshape(x, (m, n))
    Ξ = reshape(ξ, (m, n))
    return check_tangent_vector(M, X, Ξ)
end

# Display
# ---
show(io::IO, M::FixedRank) = print(io, M.abbreviation)
