using StructuredProximalOperators
using LinearAlgebra
using Test

include("check_manifold.jl")

@testset "StructuredProximalOperators.jl" begin

    ## Testing manifolds
    # @testset "Manifold $M" for M in [
    #     l1Manifold{5}([0, 0, 1, 1, 1]),
    #     FixedRankMatrices{5,6,3,ℝ}(),
    # ]
    #     check_manifold(M)

    #     check_retraction(M)
    # end


    ## Testing regularizers
    include("regularizer_l1.jl")
    include("regularizer_lnuclear.jl")


end
