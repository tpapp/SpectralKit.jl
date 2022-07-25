####
#### Chebyshev on [-1,1]
####

@testset "Chebyshev" begin
    @test_throws ArgumentError Chebyshev(InteriorGrid(), 0)
    @test_throws ArgumentError Chebyshev(EndpointGrid(), 0)
    @test_throws TypeError Chebyshev(:invalid_grid, 10)

    for grid_kind in (InteriorGrid(), EndpointGrid())
        for N in (grid_kind ≡ InteriorGrid() ? 1 : 2):10

            basis = Chebyshev(grid_kind, N)
            @test is_function_basis(basis)
            @test is_function_basis(typeof(basis))
            @test dimension(basis) == N

            # check linear combinations
            for i in 1:100
                x = rand_pm1(i)
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

            # check grid
            g = @inferred grid(basis)
            @test length(g) == N
            if grid_kind ≡ InteriorGrid()
                @test all(x -> isapprox(chebyshev_cos(x, N + 1), 0, atol = 1e-14), g)
            else
                @test all(x -> isapprox(chebyshev_cos_deriv(x, N), 0, atol = 1e-13),
                          g[2:(end-1)])
                @test g[1] == -1
                @test g[end] == 1
            end

            # augmented coefficients
            for i in 1:100
                x = rand_pm1(i)
                θ = rand(N)
                destination_basis = Chebyshev(grid_kind, N + 5)
                destination_θ = augment_coefficients(basis, destination_basis, θ)
                @test linear_combination(basis, θ, x) ≈
                    linear_combination(destination_basis, destination_θ, x)
            end

        end
    end

    # incompatible grids
    @testset "augment Chebyshev coefficients — errors" begin
        basis = Chebyshev(InteriorGrid(), 5)
        θ = randn(5)
        # incompatible grids
        basis2_G = Chebyshev(EndpointGrid(), 6)
        @test !is_subset_basis(basis, basis2_G)
        @test_throws ArgumentError augment_coefficients(basis, basis2_G, θ)
        # fewer dimensions
        basis2_N = Chebyshev(InteriorGrid(), 4)
        @test !is_subset_basis(basis, basis2_N)
        @test_throws ArgumentError augment_coefficients(basis, basis2_N, θ)
        # too few coefficients
        @test_throws ArgumentError augment_coefficients(basis, basis, randn(4))
    end
end
