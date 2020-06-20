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

### Checking retractions, gradient and hessians

```julia
julia> M = FixedRankMatrices(5, 6, 3)
FixedRankMatrices(5, 6, 3, ℝ)

julia> check_retraction(M)

julia> check_e2r_gradient_hessian(M)
- gradf(x) ∈ T_x M:				true
- η ∈ T_x M:					true
- Hess f(x)[η] ∈ T_x M:				true
- Hess f(x)[ξ] ∈ T_x M:				true
- ⟨Hess f(x)[ξ], η⟩ - ⟨Hess f(x)[η], ξ⟩:	0.0
(ηgradfx, ηHessf_xη) = (-1.3226415096782111, 1.50236594681146)
```

## TODOS:
- test l1Subspace, FixedMatrices
- devise test for regularizers
- check gradient / hessian of regularizers

### Wednesday 17th june:
Both manifolds are decent enough to start working with. Euclidean to riemannian conversion functions are fine for l1 and lnuclear regularizers. l1 regularizer has correct riemannian gradient / hessians, but nuclear norm regularizer proves more difficult.

WIP: get a first order development of the singular value decomposition, to be used to derive the riemannian hessian component of lnuclear regularizer. See `svd_development.jl` file.

### Monday 22nd june:
- Svd development and nuclear regularizer second order fixed.
- Slope detection implemented. Review of test threshold parameters for regression would be good.

TODO:
- properly test first and second order devs (tangent space, symmetry, slopes) for regularizers / conversion functions.