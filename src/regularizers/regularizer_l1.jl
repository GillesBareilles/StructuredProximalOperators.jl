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

∇M_g(reg::regularizer_l1, M::l1Manifold, x) = reg.λ * sign.(x)        ## TODO make this inplace.


## 2nd order
∇²M_g_ξ(::regularizer_l1, M::l1Manifold, x, ξ) = zeros(size(ξ))
