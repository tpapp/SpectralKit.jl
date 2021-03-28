using SpectralKit
using Test, DocStringExtensions, StaticArrays
using ForwardDiff: derivative

include("utilities.jl")

####
#### Chebyshev on [-1,1]
####

@testset "Chebyshev" begin
    @test_throws ArgumentError Chebyshev(0)

    for N in 1:10
        F = Chebyshev(N)
        @test is_function_family(F)
        @test is_function_family(typeof(F))

        for i in 1:100
            x = rand_in_domain(i, -1, 1)
            b = @inferred basis_at(F, x)

            @test length(b) == N
            @test eltype(b) == Float64
            @test collect(b) ≈ [chebyshev_cos(x, i) for i in 1:N]

            θ = rand(N)
            @test linear_combination(F, x, θ) ≈
                sum(chebyshev_cos(x, i) * θ for (i,θ) in enumerate(θ))
            @test derivative(x -> linear_combination(F, x, θ), x) ≈
                sum(chebyshev_cos_deriv(x, i) * θ for (i,θ) in enumerate(θ))
        end

        gi = @inferred grid(F, InteriorGrid())
        @test length(gi) == N
        @test all(x -> isapprox(chebyshev_cos(x, N + 1), 0, atol = 1e-14), gi)

        if N ≥ 2
            ge = @inferred grid(F, EndpointGrid())
            @test length(ge) == N
            @test all(x -> isapprox(chebyshev_cos_deriv(x, N), 0, atol = 1e-13),
                      ge[2:(end-1)])
            @test ge[1] == -1
            @test ge[end] == 1
        end
    end
end

# @testset "ChebyshevSemiInf" begin
#     @testset "to ∞" begin
#         F = ChebyshevSemiInf(2.0, 4.7)
#         @test repr(F) == "ChebyshevSemiInf(2.0, 4.7)"
#         test_is_function_family(F)
#         test_roots(F, 9)
#         test_augmented_extrema(F, 10)
#         test_endpoint_continuity(F, (2.0, Inf), 1:10; atol = 1e-3)
#         test_derivatives(F, 1:10)
#         test_linear_combinations.(F, TESTED_ORDERS)
#         test_basis_many.(F, TESTED_ORDERS)
#     end

#     @testset "from -∞" begin
#         F = ChebyshevSemiInf(3.0, -1.9)
#         test_roots(F, 11; atol = 1e-13)
#         test_augmented_extrema(F, 7; atol = 1e-10)
#         test_endpoint_continuity(F, (-Inf, 3.0), 1:10; atol = 1e-3)
#         test_derivatives(F, 1:10)
#         test_linear_combinations.(F, TESTED_ORDERS)
#         test_basis_many.(F, TESTED_ORDERS)
#     end

#     @test_throws ArgumentError ChebyshevSemiInf(0.0, 0.0)
# end

# @testset "ChebyshevInf" begin
#     F = ChebyshevInf(0.0, 1.0)

#     @test repr(F) == "ChebyshevInf(0.0, 1.0)"

#     test_is_function_family(F)

#     test_roots(F, 11)
#     @test roots(F, 11)[6] == 0  # precise 0

#     test_augmented_extrema(F, 11)
#     @test augmented_extrema(F, 11)[6] == 0 # precise 0

#     test_endpoint_continuity(F, (-Inf, Inf), 1:10)
#     test_derivatives(F, 1:10)
#     test_linear_combinations.(F, TESTED_ORDERS)
#     test_basis_many.(F, TESTED_ORDERS)

#     @test_throws ArgumentError ChebyshevInf(0.0, -3.0)
#     @test_throws ArgumentError ChebyshevInf(0.0, 0.0)

#     @test ChebyshevInf(0, 1.0) isa ChebyshevInf{Float64}
# end

# @testset "ChebyshevInterval" begin
#     F = ChebyshevInterval(2.0, 5)

#     @test repr(F) == "ChebyshevInterval(2.0, 5.0)"

#     test_is_function_family(F)

#     test_roots(F, 11)

#     test_augmented_extrema(F, 11)

#     test_endpoint_continuity(F, (2.0, 5.0), 1:10)

#     test_derivatives(F, 1:10)

#     test_linear_combinations.(F, TESTED_ORDERS)

#     test_basis_many.(F, TESTED_ORDERS)

#     @test_throws ArgumentError ChebyshevInterval(2.0, 1.0)
#     @test_throws ArgumentError ChebyshevInterval(2.0, 2)
#     @test_throws ArgumentError ChebyshevInterval(-Inf, Inf)

#     @test ChebyshevInterval(0, 1) isa ChebyshevInterval{Float64} # promotion
# end

# include("test_smolyak.jl")
