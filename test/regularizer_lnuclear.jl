@testset "lnuclear regularizer" begin
    x = [
        4.0 0.0 0.0 0.0 0.0 0.0
        0.0 3.0 0.0 0.0 0.0 0.0
        0.0 0.0 2.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0
    ]
    reg = regularizer_lnuclear(0.5)

    @test g(reg, x) == 4.5

    y, M = prox_αg(reg, x, 4)
    @test y == [
        2.0 0.0 0.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0
    ]
    @test M == FixedRankMatrices{5,6,2,ℝ}()

    # @test embed(M, x, ∇M_g(reg, M, y)) == [
    #     0.5 0.0 0.0 0.0 0.0 0.0
    #     0.0 0.5 0.0 0.0 0.0 0.0
    #     0.0 0.0 0.0 0.0 0.0 0.0
    #     0.0 0.0 0.0 0.0 0.0 0.0
    #     0.0 0.0 0.0 0.0 0.0 0.0
    # ]

    # hessxξ = ∇²M_g_ξ(reg, M, y, rand(5, 6))

    # @test norm(hessxξ[3:end, 3:end]) == 0

    comparison = check_regularizer_gradient_hessian(M, reg)
    for (k, slope) in [(:frgrad, 2), (:frhess, 3)]
        regressiondata = remove_small_functionvals(comparison[k])
        @test length(regressiondata) != 0
        slope, residual = get_sloperesidual(regressiondata)
        @test isapprox(slope, slope; atol = 1e-1)
        @test isapprox(residual, 0; atol = 5e-2)
    end
end
