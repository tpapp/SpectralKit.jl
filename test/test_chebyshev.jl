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
