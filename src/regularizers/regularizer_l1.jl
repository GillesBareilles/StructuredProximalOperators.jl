##
## L1 regularization
##
struct regularizer_l1 <: Regularizer
    λ::Float64
end


## 0th order
(reg::regularizer_l1)(x) = reg.λ * norm(x, 1)

## 1st order
function prox_αg(reg::regularizer_l1, x, α)
    res = softthresh.(x, reg.λ * α)
    return res, l1Manifold(map(x -> abs(x) > 0, res))
end

∇M_g!(reg::regularizer_l1, ::l1Manifold, ∇M_g, x) = (@. ∇M_g = reg.λ * sign.(x))
function ∇M_g(reg::regularizer_l1, M::l1Manifold, x)
    ∇M_g = zeros(representation_size(M))
    return ∇M_g!(reg, M, ∇M_g, x)
end


## 2nd order
∇²M_g_ξ(::regularizer_l1, M::l1Manifold, x, ξ) = zeros(size(ξ))
∇²M_g_ξ!(::regularizer_l1, M::l1Manifold, res, x, ξ) = (res .= 0)
