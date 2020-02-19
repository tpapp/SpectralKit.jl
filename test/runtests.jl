using SpectralKit
using Test, DocStringExtensions
import ForwardDiff

include("utilities.jl")

const TESTED_ORDERS = (Order(0), Order(1), OrdersTo(1))

####
#### tests
####

@testset "order specs" begin
    @test Order(0) ≡ Order{0}()
    @test OrdersTo(0) ≡ OrdersTo{0}()
    @test_throws ArgumentError Order(-9)
    @test_throws ArgumentError OrdersTo(-9)
end

@testset "Chebyshev" begin
    F = Chebyshev()

    @test repr(F) == "Chebyshev()"

    test_is_function_family(F)

    test_roots(F, 11)
    @test roots(F, 11)[6] == 0  # precise 0

    test_augmented_extrema(F, 11)
    @test augmented_extrema(F, 11)[6] == 0 # precise 0

    test_endpoint_continuity(F, (-1, 1), 1:10)

    test_derivatives(F, 1:10)

    test_linear_combinations.(F, TESTED_ORDERS)

    @test_throws ArgumentError basis_function(F, 0, 0.0, Order(0)) # K ≥ 1
end

@testset "ChebyshevSemiInf" begin
    @testset "to ∞" begin
        F = ChebyshevSemiInf(2.0, 4.7)
        @test repr(F) == "ChebyshevSemiInf(2.0, 4.7)"
        test_is_function_family(F)
        test_roots(F, 9)
        test_augmented_extrema(F, 10)
        test_endpoint_continuity(F, (2.0, Inf), 1:10; atol = 1e-3)
        test_derivatives(F, 1:10)
        test_linear_combinations.(F, TESTED_ORDERS)
    end

    @testset "from -∞" begin
        F = ChebyshevSemiInf(3.0, -1.9)
        test_roots(F, 11; atol = 1e-13)
        test_augmented_extrema(F, 7; atol = 1e-10)
        test_endpoint_continuity(F, (-Inf, 3.0), 1:10; atol = 1e-3)
        test_derivatives(F, 1:10)
        test_linear_combinations.(F, TESTED_ORDERS)
    end

    @test_throws ArgumentError ChebyshevSemiInf(0.0, 0.0)
end

@testset "ChebyshevInf" begin
    F = ChebyshevInf(0.0, 1.0)

    @test repr(F) == "ChebyshevInf(0.0, 1.0)"

    test_is_function_family(F)

    test_roots(F, 11)
    @test roots(F, 11)[6] == 0  # precise 0

    test_augmented_extrema(F, 11)
    @test augmented_extrema(F, 11)[6] == 0 # precise 0

    test_endpoint_continuity(F, (-Inf, Inf), 1:10)
    test_derivatives(F, 1:10)
    test_linear_combinations.(F, TESTED_ORDERS)

    @test_throws ArgumentError ChebyshevInf(0.0, -3.0)
    @test_throws ArgumentError ChebyshevInf(0.0, 0.0)

    @test ChebyshevInf(0, 1.0) isa ChebyshevInf{Float64}
end

@testset "ChebyshevInterval" begin
    F = ChebyshevInterval(2.0, 5)

    @test repr(F) == "ChebyshevInterval(2.0, 5.0)"

    test_is_function_family(F)

    test_roots(F, 11)

    test_augmented_extrema(F, 11)

    test_endpoint_continuity(F, (2.0, 5.0), 1:10)

    test_derivatives(F, 1:10)

    test_linear_combinations.(F, TESTED_ORDERS)

    @test_throws ArgumentError ChebyshevInterval(2.0, 1.0)
    @test_throws ArgumentError ChebyshevInterval(2.0, 2)
    @test_throws ArgumentError ChebyshevInterval(-Inf, Inf)

    @test ChebyshevInterval(0, 1) isa ChebyshevInterval{Float64} # promotion
end
