# StructuredProximalOperators

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://GillesBareilles.github.io/StructuredProximalOperators.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://GillesBareilles.github.io/StructuredProximalOperators.jl/dev)
[![Build Status](https://travis-ci.com/GillesBareilles/StructuredProximalOperators.jl.svg?branch=master)](https://travis-ci.com/GillesBareilles/StructuredProximalOperators.jl)
[![Coverage](https://codecov.io/gh/GillesBareilles/StructuredProximalOperators.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/GillesBareilles/StructuredProximalOperators.jl)

Implement useful tools for handling non-smooth regularizers, l1 and nuclear norm for now. We implement the following methods

```julia
julia> using StructuredProximalOperators
julia> g = regularizer_lnuclear();
julia> x = rand(7, 8);

julia> g(x)
7.0841856836292365

julia> prox_αg(g, x, 1.0);

julia> y, M = prox_αg(g, x, 1.0); M
FixedRankMatrices(7, 8, 2, ℝ)

julia> ∇M_g(g, M, y);

julia> ∇²M_g_ξ(g, M, x, ξ); #For some tangent vector ξ.
```

The manifolds try to stick to the API of `Manifolds.jl`, as defined [here](https://juliamanifolds.github.io/Manifolds.jl/latest/interface.html).

## TODOS:
- test l1Subspace, FixedMatrices
- devise test for regularizers
- check gradient / hessian of regularizers

## mercredi 17 juin:
Both manifolds are decent enough to start working with. Euclidean to riemannian conversion functions are fine for l1 and lnuclear regularizers. l1 regularizer has correct riemannian gradient / hessians, but nuclear norm regularizer proves more difficult.

WIP: get a first order development of the singular value decomposition, to be used to derive the riemannian hessian component of lnuclear regularizer. See `svd_development.jl` file.