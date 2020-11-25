##
## L1 regularization
##
@with_kw mutable struct regularizer_l1 <: Regularizer
    λ::Float64
end

function wholespace_manifold(reg::regularizer_l1, x)
    return l1Manifold(ones(size(x)))
end

## 0th order
g(reg::regularizer_l1, x) = reg.λ * norm(x, 1)

## 1st order
function prox_αg!(reg::regularizer_l1, res, x, α)
    res .= softthresh.(x, reg.λ * α)
    return l1Manifold(map(x -> abs(x) > 0, res))
end
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



#
### Optimality status
#
function model_g_subgradient!(model, regularizer::regularizer_l1, M, x)
    n = size(x, 1)
    λ = regularizer.λ

    d = manifold_dimension(M)

    @variable(model, ḡ_normal[1:n-d])
    @constraint(model, -λ .<= ḡ_normal .<= λ)

    return ḡ_normal
end

function build_subgradient_from_normalcomp(regularizer::regularizer_l1, M, x, ḡ_normal)
    n = size(x, 1)
    ḡ = Vector(undef, n)
    λ = regularizer.λ
    ind_norm = 1
    for i in 1:n
        if M.nnz_coords[i]
            ḡ[i] = sign(x[i]) * λ
        else
            ḡ[i] = ḡ_normal[ind_norm]
            ind_norm += 1
        end
    end
    return ḡ
end

function build_normalcomp_from_subgradient(::regularizer_l1, M, x, ḡ)
    return ḡ[.!M.nnz_coords]
end