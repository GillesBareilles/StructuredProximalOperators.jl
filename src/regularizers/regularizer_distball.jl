##
## distance to ||⋅||_p ball of radius r
##
struct regularizer_distball <: Regularizer
    λ::Float64
    p::Float64
    r::Float64
end

function wholespace_manifold(reg::regularizer_distball, x)
    return Euclidean(size(x)...)
end

## 0th order
g(reg::regularizer_distball, x) = reg.λ * max(0.0, norm(x, reg.p) - reg.r)


## 1st order
function prox_αg!(reg::regularizer_distball, res, x, α)
    normxp = norm(x, reg.p)
    M = Euclidean(size(x)...)

    if normxp ≤ reg.r
        res .= x
    elseif reg.r < normxp ≤ reg.r + α
        res .= reg.r .* x ./ normxp
        M = Sphere{p,r}(size(x)...)
    else
        res .= x .* (1 - α / normxp)
    end

    return M
end
function prox_αg(reg::regularizer_distball, x, α)
    res = zero(x)
    M = prox_αg!(reg, res, x, α)
    return res, M
end

# ∇M_g!(reg::regularizer_distball, ::l1Manifold, ∇M_g, x) = (@. ∇M_g = reg.λ * sign.(x))
# function ∇M_g(reg::regularizer_distball, M::l1Manifold, x)
#     ∇M_g = zeros(representation_size(M))
#     return ∇M_g!(reg, M, ∇M_g, x)
# end


# ## 2nd order
# ∇²M_g_ξ(::regularizer_distball, M::l1Manifold, x, ξ) = zeros(size(ξ))
# ∇²M_g_ξ!(::regularizer_distball, M::l1Manifold, res, x, ξ) = (res .= 0)
