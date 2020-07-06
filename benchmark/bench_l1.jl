using BenchmarkTools
using StructuredProximalOperators
using Random

function main()

    benchGVal = true
    benchProx = true
    benchGrad = true
    benchHess = true

    g = regularizer_l1(2.5)
    n = 1000000
    Random.seed!(1567)
    x = rand(n).*5

    ## Prox
    y, M = prox_αg(g, x, 1.0)
    @show M

    if benchGVal
        println("\n + g(y)")
        g(y)
        @btime $g($y)
    end

    if benchProx
        println("\n + Prox computation")
        println("   - inplace")
        res = zeros(n)
        prox_αg!(g, res, x, 1.0)
        @btime prox_αg!($g, $res, $x, 1.0)

        println("   - with allocation")
        prox_αg(g, x, 1.0)
        @btime prox_αg($g, $x, 1.0)
    end

    ## Gradient, hessian
    if benchGrad
        println("\n + Gradient computation")
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
        println("\n + Hessian-vector computation")
        println("   - inplace")
        d = zeros(n)
        res = zeros(n)
        ∇²M_g_ξ!(g, M, res, x, d)
        @btime ∇²M_g_ξ!($g, $M, $res, $x, $d);

        println("   - with allocation")
        ∇²M_g_ξ(g, M, x, d)
        @btime ∇²M_g_ξ!($g, $M, $res, $x, $d);
    end


    return
end

main()
