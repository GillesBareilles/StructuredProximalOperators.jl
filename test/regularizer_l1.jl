@testset "l1 regularizer" begin
    x = [1, 2, 3, 4, 5]
    g = regularizer_l1(0.5)

    @test g(x) == 7.5

    y, M = prox_αg(g, x, 4.5)
    @test y == [0.0, 0.0, 0.75, 1.75, 2.75]
    @test M == l1Manifold{5}([0, 0, 1, 1, 1])

    @test ∇M_g(g, M, y) == [0.0, 0.0, 0.5, 0.5, 0.5]

    @test norm(∇²M_g_ξ(g, M, y, rand(5))) == 0

    comparison = check_regularizer_gradient_hessian(M, g)
    for (k, slope) in [(:frgrad, 2), (:frhess, 3)]
        regressiondata = remove_small_functionvals(comparison[k])
        @test length(regressiondata) == 0
    end
end
