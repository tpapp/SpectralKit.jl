####
#### Chebyshev on [-1,1]
####

@testset "Chebyshev" begin
    @test_throws ArgumentError Chebyshev(InteriorGrid(), 0)
    @test_throws ArgumentError Chebyshev(EndpointGrid(), 1)

    for grid_kind in (InteriorGrid(), EndpointGrid())
        for N in (grid_kind ≡ InteriorGrid() ? 1 : 2):10

            basis = Chebyshev(grid_kind, N)
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
        end
    end
end
