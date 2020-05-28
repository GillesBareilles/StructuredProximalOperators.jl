module StructuredProximalOperators

# Write your package code here.
using LinearAlgebra
using Manifolds

import Base: show, ==

## Manifolds exports
export manifold_dimension
export randomMPoint, randomTVector
export inner, norm
export project!, project
export retract!, retract
export egrad_to_rgrad!, egrad_to_rgrad, ehess_to_rhess!, ehess_to_rhess
export check_manifold_point, check_tangent_vector, show

## Regularizers exports
export prox_αg, ∇M_g, ∇²M_g_ξ
export regularizer_l1
export regularizer_lnuclear


abstract type AbstractRegularizer end
const AbstractManifold = Manifolds.Manifold{ℝ}


## Helper
softthresh(x, α) = sign(x) * max(0, abs(x) - α)


## Manifolds
include("manifolds/l1subspace.jl")
include("manifolds/FixedRankMatrices.jl")

## Regularizers
include("regularizers/regularizer_l1.jl")
include("regularizers/regularizer_lnuclear.jl")

end
