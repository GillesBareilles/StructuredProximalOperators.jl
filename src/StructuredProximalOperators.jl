module StructuredProximalOperators

using LinearAlgebra
using ManifoldsBase
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
export l1Manifold, regularizer_l1
export FixedRank, regularizer_lnuclear


abstract type Regularizer end


## Helper
softthresh(x, α) = sign(x) * max(0, abs(x) - α)


##
egrad_to_rgrad!(M::Manifold, gradf_x, x, ∇f_x) = project!(M, gradf_x, x, ∇f_x)
egrad_to_rgrad(M::Manifold, x, ∇f_x) = project(M, x, ∇f_x)
function ehess_to_rhess(M::Manifold, x, ∇f_x, ∇²f_ξ, ξ)
    Hessf_xξ = Manifolds.allocate(ξ)
    return ehess_to_rhess!(M, Hessf_xξ, x, ∇f_x, ∇²f_ξ, ξ)
end


## Manifolds
include("manifolds/l1subspace.jl")
include("manifolds/FixedRankMatrices.jl")

## Regularizers
include("regularizers/regularizer_l1.jl")
include("regularizers/regularizer_lnuclear.jl")

end
