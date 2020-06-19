function check_manifold(M)
    # check random points, vectors
    for i in 1:10
        x = randomMPoint(M)
        @test is_manifold_point(M, x)

        ξ = randomTVector(M, x)
        @test is_tangent_vector(M, x, ξ, atol = 1e-10)
        @test isapprox(norm(M, x, ξ), 1)

        # check projection
        q = rand(representation_size(M)...)

        η = zero_tangent_vector(M, x)
        project!(M, η, x, q)
        @test is_tangent_vector(M, x, η; atol = 1e-10)

        # check retraction
        y = retract(M, x, ξ)
        @test is_manifold_point(M, y)
    end

    return nothing
end
