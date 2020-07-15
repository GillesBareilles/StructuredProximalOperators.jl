##
## L1 regularization
##
@with_kw mutable struct regularizer_group{Tr<:Regularizer} <: Regularizer
    groups::Vector{UnitRange{Int64}}
    regs::Vector{Tr}
end

function <(M1::ProductManifold, M2::ProductManifold)
    return M1.manifolds < M2.manifolds
end

function wholespace_manifold(reg::regularizer_group{Tr}, x) where {Tr}
    return ProductManifold([
        wholespace_manifold(reg.regs[i], x[group]) for (i, group) in enumerate(reg.groups)
    ]...)
end


## 0th order
function g(reg::regularizer_group, x)
    return sum(g(reg.regs[i], x[group]) for (i, group) in enumerate(reg.groups))
end

## 1st order
function prox_αg!(reg::regularizer_group, res, x, α)
    Ms = []
    for (i, group) in enumerate(reg.groups)
        push!(Ms, prox_αg!(reg.regs[i], view(res, group), view(x, group), α))
    end
    return ProductManifold(Ms...)
end


# ∇M_g!(reg::regularizer_groupdistball, ::l1Manifold, ∇M_g, x) = (@. ∇M_g = reg.λ * sign.(x))
# function ∇M_g(reg::regularizer_groupdistball, M::l1Manifold, x)
#     ∇M_g = zeros(representation_size(M))
#     return ∇M_g!(reg, M, ∇M_g, x)
# end


# ## 2nd order
# ∇²M_g_ξ(::regularizer_groupdistball, M::l1Manifold, x, ξ) = zeros(size(ξ))
# ∇²M_g_ξ!(::regularizer_groupdistball, M::l1Manifold, res, x, ξ) = (res .= 0)
