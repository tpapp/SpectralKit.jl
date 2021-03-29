using SpectralKit
using Test, DocStringExtensions, StaticArrays
using ForwardDiff: derivative

using SpectralKit: to_domain, from_domain

include("utilities.jl")

####
#### Chebyshev on [-1,1]
####

@testset "Chebyshev" begin
    @test_throws ArgumentError Chebyshev(0)

    for N in 1:10
        basis = Chebyshev(N)
        @test is_function_basis(basis)
        @test is_function_basis(typeof(basis))
        @test dimension(basis) == N

        for i in 1:100
            x = rand_in_domain(i, -1, 1)
            bx = @inferred basis_at(basis, x)

            @test length(bx) == N
            @test eltype(bx) == Float64
            @test collect(bx) ≈ [chebyshev_cos(x, i) for i in 1:N]

            θ = rand(N)
            @test linear_combination(basis, θ, x) ≈
                sum(chebyshev_cos(x, i) * θ for (i,θ) in enumerate(θ))
            @test derivative(linear_combination(basis, θ), x) ≈
                sum(chebyshev_cos_deriv(x, i) * θ for (i,θ) in enumerate(θ))
        end

        gi = @inferred grid(basis, InteriorGrid())
        @test length(gi) == N
        @test all(x -> isapprox(chebyshev_cos(x, N + 1), 0, atol = 1e-14), gi)

        if N ≥ 2
            ge = @inferred grid(basis, EndpointGrid())
            @test length(ge) == N
            @test all(x -> isapprox(chebyshev_cos_deriv(x, N), 0, atol = 1e-13),
                      ge[2:(end-1)])
            @test ge[1] == -1
            @test ge[end] == 1
        end
    end
end

@testset "Chebyshev semi-infinite" begin
    f(x) = exp(-3*x)
    f′(x) = -3*f(x)
    grid_dense = range(3.1, 8, length = 100)

    A = 3.0
    L = 4.0
    N = 11
    basis0 = Chebyshev(N)
    trans = SemiInfRational(A, L)
    basis = univariate_basis(basis0, trans)
    @test dimension(basis) == N
    @test domain(basis) == (A, Inf)

    @testset "transformation" begin
        for i in 1:100
            x = rand_in_domain(i, A, Inf)
            @test from_domain(trans, basis0, to_domain(trans, basis0, x)) ≈ x
        end
    end

    @testset "interior grid" begin
        gi = grid(basis, InteriorGrid())
        @test length(gi) == N
        @test sum(gi .< A + L) == 5
        @test all(A .< gi .< Inf)
        C = collocation_matrix(basis, gi)
        θ = C \ f.(gi)
        @test maximum(x -> abs(linear_combination(basis, θ, x) - f(x)), grid_dense) ≤ 1e-8
        @test maximum(x -> abs(derivative(linear_combination(basis, θ), x) - f′(x)),
                      grid_dense) ≤ 1e-7
    end

    @testset "endpoints grid" begin
        ge = grid(basis, EndpointGrid())
        @test length(ge) == N
        @test sum(ge .< A + L) == 5
        @test all(A .≤ ge .≤ Inf)
        C = collocation_matrix(basis, ge)
        θ = C \ f.(ge)
        @test maximum(x -> abs(linear_combination(basis, θ, x) - f(x)), grid_dense) ≤ 1e-8
        @test maximum(x -> abs(derivative(linear_combination(basis, θ), x) - f′(x)),
                      grid_dense) ≤ 1e-7
    end
end

@testset "Chebyshev infinite" begin
    f(x) = exp(-4*abs2(x))
    f′(x) = -8*x*f(x)
    grid_dense = range(-1, 2, length = 100)

    A = 0.0
    L = 1.0
    N = 20
    basis0 = Chebyshev(N)
    trans = InfRational(A, L)
    basis = univariate_basis(basis0, trans)
    @test dimension(basis) == N
    @test domain(basis) == (-Inf, Inf)

    @testset "transformation" begin
        for i in 1:100
            x = rand_in_domain(i, -Inf, Inf)
            @test from_domain(trans, basis0, to_domain(trans, basis0, x)) ≈ x
        end
    end

    @testset "interior grid" begin
        gi = grid(basis, InteriorGrid())
        @test length(gi) == N
        @test sum(gi .< A) == 10
        @test all(-Inf .< gi .< Inf)
        C = collocation_matrix(basis, gi)
        θ = C \ f.(gi)
        @test maximum(x -> abs(linear_combination(basis, θ, x) - f(x)), grid_dense) ≤ 1e-4
        @test maximum(x -> abs(derivative(linear_combination(basis, θ), x) - f′(x)),
                      grid_dense) ≤ 2e-3
    end

    @testset "endpoints grid" begin
        ge = grid(basis, EndpointGrid())
        @test length(ge) == N
        @test sum(ge .< A) == 10
        @test all(-Inf .≤ ge .≤ Inf)
        C = collocation_matrix(basis, ge)
        θ = C \ f.(ge)
        @test maximum(x -> abs(linear_combination(basis, θ, x) - f(x)), grid_dense) ≤ 1e-4
        @test maximum(x -> abs(derivative(linear_combination(basis, θ), x) - f′(x)),
                      grid_dense) ≤ 2e-3
    end
end

# @pgf Axis({ no_marks },
#           Plot(Table(grid_dense, f′.(grid_dense))),
#           Plot(Table(grid_dense, (x -> derivative(linear_combination(basis, θ), x)).(grid_dense)))
#           )

# @testset "ChebyshevSemiInf" begin
#     @testset "to ∞" begin
#         F = ChebyshevSemiInf(2.0, 4.7)
#         @test repr(F) == "ChebyshevSemiInf(2.0, 4.7)"
#         test_is_function_basis(F)
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

#     test_is_function_basis(F)

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

#     test_is_function_basis(F)

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
