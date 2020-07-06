using BenchmarkTools
using StructuredProximalOperators
using Random

function main()

    benchGVal = true
    benchProx = false
    benchGrad = false
    benchHess = false

    g = regularizer_lnuclear(2.5)
    m, n = 200, 300
    Random.seed!(1567)
    x = rand(m, n)

    ## Prox
    y, M = prox_αg(g, x, 2.5)
    @show M
    @show typeof(y)

    if benchGVal
        println(" + g(y)")

        g(y)
        @btime $g($y)
    end

    if benchProx
        println(" + Prox computation")
        println("   - inplace")
        res = zeros(m, n)
        prox_αg!(g, res, y, 1.0)
        @btime prox_αg!($g, $res, $y, 1.0)

        println("   - with allocation")
        prox_αg(g, y, 1.0)
        @btime prox_αg($g, $y, 1.0)
    end

    ## Gradient, hessian
    if benchGrad
        println(" + Gradient computation")
        println("   - inplace")
        res = zeros(n)
        ∇M_g!(g, M, res, x)
        @btime ∇M_g!($g, $M, $res, $x);

        println("   - with allocation")
        ∇M_g(g, M, x)
        @btime ∇M_g($g, $M, $x);
    end

    ## Gradient / hessian euclidean to riemannian
    if benchHess
        println(" + Hessian-vector computation")
        println("   - inplace")
        d = zeros(n)
        res = zeros(n)
        ∇²M_g_ξ!(g, M, res, x, d)
        @btime ∇²M_g_ξ!($g, $M, $res, $x, $d);

        println("   - with allocation")
        ∇²M_g_ξ(g, M, x, d)
        @btime ∇²M_g_ξ($g, $M, $x, $d);
    end


    return
end

main()
