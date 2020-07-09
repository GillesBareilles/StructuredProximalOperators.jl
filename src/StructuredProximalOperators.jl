module StructuredProximalOperators

import LinearAlgebra: norm
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
using ManifoldsBase
using Manifolds

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

## Regularizers exports
export g, prox_αg, prox_αg!, ∇M_g, ∇M_g!, ∇²M_g_ξ, ∇²M_g_ξ!

export l1Manifold, regularizer_l1
export FixedRankMatrices, regularizer_lnuclear
export PSphere, regularizer_distball

export wholespace_manifold

export copy
export Regularizer

abstract type Regularizer end

<(::Euclidean, ::Euclidean) = false



function show(io::IO, M::Euclidean)
    return print(io, name(M))
end
function name(M::Euclidean{dim, 𝔽}; short = true) where {dim, 𝔽}
    rep = representation_size(M)
    if length(rep)==1
        return "$𝔽 ^"*string(rep[1])
    else
        return "$𝔽 ^"*string(rep)
    end
end


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
include("manifolds/FixedRankMatrices.jl")
include("manifolds/PShpere.jl")

## Regularizers
include("regularizers/regularizer_l1.jl")
include("regularizers/regularizer_lnuclear.jl")
include("regularizers/regularizer_distball.jl")


include("compare_smoothcurves.jl")

export compare_curves,
    display_curvescomparison,
    check_e2r_gradient_hessian,
    check_retraction,
    check_regularizer_gradient_hessian

end
