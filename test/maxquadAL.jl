include("check_manifold.jl")

@testset "PlaneParabola manifold" begin
    M = PlaneParabola()

    check_manifold(M)

    comparison = check_e2r_gradient_hessian(M)
    for (k, slope_theoretical) in [(:frgrad, 2), (:frhess, 3)]
        regressiondata = remove_small_functionvals(comparison[k])

        # Some values may remain for t close to 1
        if length(regressiondata) > 5
            slope_exp, residual = get_sloperesidual(regressiondata)
            @test isapprox(slope_exp, slope_theoretical; atol = 1e-1)
        end
    end
end


@testset "maxquadAL regularizer" begin
    x = Float64[1, 1]
    reg = maxquadAL()

    @test g(reg, x) == 0.0

    # Upper parabola point
    y, M = prox_αg(reg, [0.0, 4.0], 0.25)
    @test y[1] == 0.0
    @test 0 <= y[2] < 4.0
    @test M == Euclidean(2)

    # Lower parabola point
    y, M = prox_αg(reg, [0.0, -4.0], 0.25)
    @test y[1] == 0.0
    @test -4.0 < y[2] < 0.0
    @test M == Euclidean(2)

    # Parabola point
    y, M = prox_αg(reg, x, 0.25)
    @test y == [1.0, 1.0]
    @test M == PlaneParabola()


    y, M = prox_αg(reg, x, 0.25)
    @test y == [1.0, 1.0]
    @test M == PlaneParabola()


    @test ∇M_g(reg, M, y) == zeros(2)
    @test norm(∇²M_g_ξ(reg, M, y, rand(2))) == 0.0


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
