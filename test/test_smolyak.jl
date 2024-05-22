using SpectralKit, Test
using SpectralKit: PM1

####
#### api
####

@testset "Smolyak API checks" begin
    @test_throws MethodError smolyak_basis(Chebyshev, :invalid_grid, SmolyakParameters(3), 2)
    @test_logs (:warn, "M > B replaced with M = B") SmolyakParameters(2, 4)
end

@testset "Smolyak API sanity checks" begin
    f(x) = (x[1] - 3) * (x[2] + 5) # linear function, just a sanity check
    basis = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(3), 2)
    @test @inferred(domain(basis)) ≡ coordinate_domains(PM1(), PM1())
    g = grid(Float64, basis)
    iterator_sanity_checks(g)
    x = @inferred collect(g)
    M = @inferred collocation_matrix(basis, x)
    θ = M \ f.(x)
    @test sum(abs.(θ) .> 1e-8) == 4
    y1 = range(extrema(domain(basis)[1])...; length = 100)
    y2 = range(extrema(domain(basis)[2])...; length = 100)
    for y1 in y1
        for y2 in y2
            y = SVector(y1, y2)
            @test linear_combination(basis, θ, y) ≈ f(y)
        end
    end
end

@testset "Smolyak API allocations" begin
    basis = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(3), 2)
    y = SVector(0.4, 0.7)
    θ = randn(dimension(basis))
    @inferred linear_combination(basis, θ, y)
    @test @ballocated(linear_combination($basis, $θ, $y)) == 0
end

###
### augment coefficients
###

@testset "Smolyak augment coefficients" begin
    basis1 = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(2, 2), 2)
    θ1 = randn(dimension(basis1))

    # grid ≠
    basis2_G = smolyak_basis(Chebyshev, EndpointGrid(), SmolyakParameters(2, 3), 2)
    @test !is_subset_basis(basis1, basis2_G)
    @test_throws ArgumentError augment_coefficients(basis1, basis2_G, θ1)

    # smolyak_parameters <
    basis2_P = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(2, 1), 2)
    @test !is_subset_basis(basis1, basis2_P)
    @test_throws ArgumentError augment_coefficients(basis1, basis2_P, θ1)

    # compatible basis
    basis2 = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(3, 2), 2)
    θ2 = @inferred augment_coefficients(basis1, basis2, θ1)
    @test length(θ2) == dimension(basis2)
    @test eltype(θ2) == eltype(θ1)
    for _ in 1:100
        x = (rand(), rand()) .* 4
        @test linear_combination(basis1, θ1, x) ≈ linear_combination(basis2, θ2, x)
    end
end

@testset "Smolyak nesting" begin
    for grid_kind in GRIDS
        for M1 in 0:5
            for M2 in (M1 + 1):5
                for B1 in 0:M1
                    for B2 in (B1 + 1):M2
                        basis1 = smolyak_basis(Chebyshev, grid_kind, SmolyakParameters(B1, M1), 2)
                        basis2 = smolyak_basis(Chebyshev, grid_kind, SmolyakParameters(B2, M2), 2)
                        @test is_approximate_subset(collect(grid(basis1)), collect(grid(basis2)))
                    end
                end
            end
        end
    end
end
