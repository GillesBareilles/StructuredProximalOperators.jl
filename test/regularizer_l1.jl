@testset "l1 regularizer" begin
    x = [1, 2, 3, 4, 5]
    reg = regularizer_l1(0.5)

    @test g(reg, x) == 7.5

    y, M = prox_αg(reg, x, 4.5)
    @test y == [0.0, 0.0, 0.75, 1.75, 2.75]
    @test M == l1Manifold{5}([0, 0, 1, 1, 1])

    @test ∇M_g(reg, M, y) == [0.0, 0.0, 0.5, 0.5, 0.5]

    @test norm(∇²M_g_ξ(reg, M, y, rand(5))) == 0

    comparison = check_regularizer_gradient_hessian(M, reg)
    for (k, slope) in [(:frgrad, 2), (:frhess, 3)]
        regressiondata = remove_small_functionvals(comparison[k])

        # Some vaues may remain for t close to 1
        if length(regressiondata) > 5
            slope, residual = get_sloperesidual(regressiondata)
            @test isapprox(slope, slope; atol = 1e-1)
            @test isapprox(residual, 0; atol = 5e-2)
        end
    end
end
