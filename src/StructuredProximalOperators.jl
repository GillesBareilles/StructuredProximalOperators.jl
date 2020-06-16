module StructuredProximalOperators

import LinearAlgebra: norm
import Manifolds:
    check_manifold_point,
    check_tangent_vector,
    distance,
    exp!,
    inner,
    log!,
    manifold_dimension,
    project!,
    representation_size,
    retract
using LinearAlgebra
using ManifoldsBase
using Manifolds


using PGFPlotsX, LaTeXStrings, ForwardDiff

import Base: show, ==

## Manifolds exports
export manifold_dimension
export randomMPoint, randomTVector
export inner, norm
export project!, project
export retract!, retract
export egrad_to_rgrad!, egrad_to_rgrad, ehess_to_rhess!, ehess_to_rhess
export check_manifold_point, check_tangent_vector, show
export is_manifold_point, is_tangent_vector
export representation_size
export embed!, embed

## Regularizers exports
export prox_αg, ∇M_g, ∇²M_g_ξ
export l1Manifold, regularizer_l1
export FixedRankMatrices, regularizer_lnuclear, ℝ


abstract type Regularizer end


## Helper
softthresh(x, α) = sign(x) * max(0, abs(x) - α)


##
egrad_to_rgrad!(M::Manifold, gradf_x, x, ∇f_x) = project!(M, gradf_x, x, ∇f_x)
function egrad_to_rgrad(M::Manifold, x, ∇f_x)
    gradf = zero_tangent_vector(M, x)
    return project!(M, gradf, x, ∇f_x)
end
function ehess_to_rhess(M::Manifold, x, ∇f_x, ∇²f_ξ, ξ)
    Hessf_xξ = Manifolds.allocate(ξ)
    return ehess_to_rhess!(M, Hessf_xξ, x, ∇f_x, ∇²f_ξ, ξ)
end


## Manifolds
include("manifolds/l1subspace.jl")

#FixedRankMatrices

function randomMPoint(M::FixedRankMatrices{m,n,k,ℝ}) where {m,n,k}
    return SVDMPoint(rand(m, n), k)
end

function randomTVector(M::FixedRankMatrices{m,n,k,ℝ}, x) where {m,n,k}
    ξ = zero_tangent_vector(M, x)
    return project!(M, ξ, x, rand(m, n))
end

function retract(M::FixedRankMatrices{m,n,k,ℝ}, x, ξ) where {m,n,k}
    return retract(M, x, ξ, PolarRetraction())
end

embed!(::FixedRankMatrices, q, p) = (q = p.U * Diagonal(p.S) * p.Vt)
function embed(M::FixedRankMatrices{m,n,k,ℝ}, p::SVDMPoint) where {m,n,k}
    p_embed = zeros(m, n)
    return embed!(M, p_embed, p)
end

function embed!(::FixedRankMatrices, ξ_embed, p, ξ)
    return (∇²f_ξ = p.U * ξ.M * p.Vt + ξ.U * p.Vt + p.U * ξ.Vt)
end
function embed(M::FixedRankMatrices{m,n,k,ℝ}, p::SVDMPoint, ξ::UMVTVector) where {m,n,k}
    ξ_embed = zeros(m, n)
    return embed!(M, ξ_embed, p, ξ)
end

function ehess_to_rhess(
    M::FixedRankMatrices{m,n,k,ℝ},
    x::SVDMPoint,
    ∇f_x,
    ∇²f_ξ,
    ξ::UMVTVector,
) where {m,n,k}
    return UMVTVector(
        (Diagonal(ones(m)) - x.U * x.U') *
        (∇²f_ξ * x.Vt' + ∇f_x * ξ.Vt' * Diagonal(x.S .^ -1)),
        x.U' * ∇²f_ξ * x.Vt',
        (
            (Diagonal(ones(n)) - x.Vt' * x.Vt) *
            (∇²f_ξ' * x.U + ∇f_x' * ξ.U * Diagonal(x.S .^ -1))
        )',
    )
end

## Regularizers
include("regularizers/regularizer_l1.jl")
include("regularizers/regularizer_lnuclear.jl")


include("compare_smoothcurves.jl")

export compare_curves, display_curvescomparison, check_e2r_gradient_hessian

end
