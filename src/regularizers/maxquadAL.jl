##
### g(x, y) = |x²-y|
##
struct maxquadAL <: Regularizer
end

g(::maxquadAL, xy) = abs(xy[1]^2 - xy[2])

function prox_αg!(::maxquadAL, res, xy, γ)
    x̄, ȳ = xy[1], xy[2]

    ## Detect structure
    man = Euclidean(2)

    if ȳ ≤ (x̄/(1+2γ))^2 + γ
        res[1] = x̄ / (1+2γ)
        res[2] = ȳ + γ
    elseif (x̄/(1-2γ))^2 - γ ≤ ȳ
        res[1] = x̄ / (1-2γ)
        res[2] = ȳ - γ
    else
        man = PlaneParabola()

        ## Compute prox output
        a = 16*γ^2
        b = 8*γ*(1-2γ)
        c = (1-2γ)^2
        ā = -2γ
        b̄ = γ - ȳ

        ts = roots([
            c*b̄ + x̄^2,
            b*b̄ + c*ā,
            a*b̄ + b*ā,
            a*ā
        ])
        ts = filter(c -> abs(imag(c)) < 1e-12, ts)
        length(ts) == 0 && @warn "Prox of maxquadAL troubled" ts
        ts = map(c -> real(c), ts)
        length(ts) == 0 && @warn "Prox of maxquadAL troubled" ts
        ts = filter(c -> 0 <= c <= 1, ts)
        length(ts) != 1 && @warn "Prox of maxquadAL troubled" ts
        t = first(ts)

        res[1] = x̄ / (1+4γ*t-2γ)
        res[2] = ȳ + 2γ*t - γ

        !is_manifold_point(man, res) && @warn "prox_αg! returned point not on manifold"
    end

    return man
end


∇M_g!(::maxquadAL, M, ∇M_g, xy) = (∇M_g .= 0)
∇M_g(::maxquadAL, M, xy) = zeros(2)

∇²M_g_ξ(::maxquadAL, M, xy, ξ) = zeros(2)
∇²M_g_ξ!(::maxquadAL, M, res, xy, ξ) = (res .= 0)
