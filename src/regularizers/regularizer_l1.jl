##
## L1 regularization
##
struct regularizer_l1 <: AbstractRegularizer end


## 0th order
(::regularizer_l1)(x) = norm(x, 1)

## 1st order
function prox_αg(::regularizer_l1, x, α)
    res = softthresh.(x, α)
    return res, l1Manifold(map(x -> abs(x) > 0, res))
end

∇M_g(::regularizer_l1, M::l1Manifold, x) = sign.(x)


## 2nd order
∇²M_g_ξ(::regularizer_l1, M::l1Manifold, x, ξ) = zeros(size(ξ))
