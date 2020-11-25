using LinearAlgebra

@testset "lnuclear regularizer" begin
    x = [
        5.0 0.0 0.0 0.0 0.0 0.0
        0.0 4.0 0.0 0.0 0.0 0.0
        0.0 0.0 2.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0
    ]
    reg = regularizer_lnuclear(0.5)

    @test g(reg, x) == 5.5

    y, Man = prox_αg(reg, x, 4)
    @test embed(Man, y) == [
        3.0 0.0 0.0 0.0 0.0 0.0
        0.0 2.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0 0.0 0.0
    ]
    @test Man == FixedRankMatrices{5,6,2,ℝ}()





    comparison = check_regularizer_gradient_hessian(Man, reg)
    for (k, slope) in [(:frgrad, 2), (:frhess, 3)]
        regressiondata = remove_small_functionvals(comparison[k])
        @test length(regressiondata) != 0
        slope, residual = get_sloperesidual(regressiondata)
        @test isapprox(slope, slope; atol = 1e-1)
        @test isapprox(residual, 0; atol = 5e-2)
    end
end
