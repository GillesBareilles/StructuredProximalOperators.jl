module StructuredProximalOperators

using Parameters
import LinearAlgebra: norm, dot
import Manifolds:
    check_manifold_point,
    check_tangent_vector,
    distance,
    embed,
    embed!,
    exp!,
    inner,
    log!,
    manifold_dimension,
    project,
    project!,
    representation_size,
    retract,
    retract!,
    zero_tangent_vector
using LinearAlgebra
using SparseArrays

using ManifoldsBase
using Manifolds

using JuMP

using Random

using PGFPlotsX, LaTeXStrings, ForwardDiff

import Manifolds: show
import Base: show, ==, <, copy

## Manifolds exports
export Manifold
export SVDMPoint, UMVTVector
export Euclidean, ℝ
export manifold_dimension
export randomMPoint, randomTVector
export inner, norm
export project, project!
export retract, retract!
export egrad_to_rgrad!, egrad_to_rgrad, ehess_to_rhess!, ehess_to_rhess
export check_manifold_point, check_tangent_vector, show

export is_manifold_point, is_tangent_vector
export representation_size
export embed!, embed, zero_tangent_vector
export embedding_dimension

## Regularizers exports
export g, prox_αg, prox_αg!, ∇M_g, ∇M_g!, ∇²M_g_ξ, ∇²M_g_ξ!
export model_g_subgradient!, build_subgradient_from_normalcomp


export l1Manifold, regularizer_l1
export FixedRankMatrices, regularizer_lnuclear
export PSphere, regularizer_distball
export ProductManifold, regularizer_group

export wholespace_manifold

export copy
export Regularizer

abstract type Regularizer end

<(::Euclidean, ::Euclidean) = false

function copy(M::Euclidean{N,𝔽}) where {N,𝔽}
    return Euclidean(representation_size(M)...; field = 𝔽)
end

function copy(M::ProductManifold{𝔽,TM}) where {𝔽,TM}
    return ProductManifold(M.manifolds...)
end



function name(M::Euclidean{dim,𝔽}; short = true) where {dim,𝔽}
    rep = representation_size(M)
    res = "$𝔽 ^" * string(rep)
    if length(rep) == 1
        res = "$𝔽 ^" * string(rep[1])
    end
    return res
end


## Helper
softthresh(x, α) = sign(x) * max(0, abs(x) - α)



## Interface

function g(::Regularizer, x)
    return error("g not implemented for regularizer $(typeof(g)) and point $(typeof(x)).")
end


function prox_αg(reg::T, x, α) where {T<:Regularizer}
    res = zero(x)
    M = prox_αg!(reg, res, x, α)
    return res, M
end
function prox_αg!(g, res, x, α)
    return error("prox_αg! not implemented for regularizer $(typeof(g)), point $(typeof(x)).")
end

function ∇M_g(g, M, x)
    return error("∇M_g not implemented for regularizer $(typeof(g)), manifold $M, point $(typeof(x)).")
end
function ∇M_g!(g, M, res, x)
    return error("∇M_g! not implemented for regularizer $(typeof(g)), manifold $M, result $(typeof(res)), point $(typeof(x)).")
end

function ∇²M_g_ξ(g, M, x, ξ)
    return error("∇²M_g_ξ not implemented for regularizer $(typeof(g)), manifold $M, point $(typeof(x)), vector $(typeof(ξ)).")
end
function ∇²M_g_ξ!(g, M, res, x, ξ)
    return error("∇²M_g_ξ not implemented for regularizer $(typeof(g)), manifold $M, result $(typeof(res)), point $(typeof(x)), vector $(typeof(ξ)).")
end



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

function wholespace_manifold(::Tr, x) where {Tr<:Regularizer}
    return Euclidean(size(x)...)
end

## Manifolds
include("manifolds/l1subspace.jl")
include("manifolds/FixedRankMatrices.jl")
include("manifolds/PShpere.jl")

## Regularizers
include("regularizers/regularizer_l1.jl")
include("regularizers/regularizer_lnuclear.jl")
include("regularizers/regularizer_distball.jl")
include("regularizers/regularizer_group.jl")


include("compare_smoothcurves.jl")

export compare_curves,
    display_curvescomparison,
    check_e2r_gradient_hessian,
    check_retraction,
    check_regularizer_gradient_hessian

end
