####
#### Chebyshev on [-1,1]
####

"""
For testing Chebyshev blocks and shuffle.
"""
function _shuffle(len::Int; endpoints = true)
    if len == 1
        return [[1]]
    end
    if !endpoints
        len += 2            # for Interior, we start the algorithm with two extra points
    end
    shuffle = [[len ÷ 2 + 1], [1, len]]
    while sum(length, shuffle) < len
        used = sort!(reduce(vcat, shuffle))
        s = Int[]
        for i in 1:(length(used)-1)
            a = used[i]
            b = used[i + 1]
            c = (a + b) ÷ 2
            @assert a < c < b
            push!(s, c)
        end
        push!(shuffle, s)
    end
    if !endpoints
        deleteat!(shuffle, 2)
    end
    shuffle
end

@testset "blocks and shuffle" begin
    for kind in [Interior(), Endpoints()]
        for level in 1:5
            N = SK.grid_length(Chebyshev(), kind, level)
            block_lengths = map(level -> SK.block_length(Chebyshev(), kind, level), 1:level)
            @test sum(block_lengths) == N
            S = _shuffle(N; endpoints = kind == Endpoints())
            @test block_lengths == map(length, S)
        end
    end
end


@testset "Chebyshev grid" begin
    for kind in [Interior(), Endpoints()]
        previous = Float64[]
        for level in 1:7
            b = univariate_basis(Chebyshev(), kind, BoundedLinear(1.0, 3.0), level)
            g = collect(grid(b))
            @test is_approximate_subset(previous, g)
            g = previous
        end
    end
end


@testset "Chebyshev basics" begin
    transformation = BoundedLinear(1.0, 3.0)
    @test_throws ArgumentError univariate_basis(Chebyshev(), Endpoints(), transformation, 0)
    for grid_kind in (Interior(), Endpoints())
        for level in 1:5
            basis = univariate_basis(Chebyshev(), grid_kind, transformation, level)
            @test is_function_basis(basis)
            @test is_function_basis(typeof(basis))
            N = @inferred dimension(basis)
            @test N > 1

            # check linear combinations
            for _ in 1:100
                x = rand_in_domain(basis)
                bx = @inferred basis_at(basis, x)

                @test length(bx) == N
                @test eltype(bx) == Float64
                y = transform_to(PM1(), transformation, x)
                @test collect(bx) ≈ [chebyshev_cos(y, i) for i in 1:N]

                θ = rand(N)
                @test linear_combination(basis, θ, x) ≈
                    sum(chebyshev_cos(y, i) * θ for (i,θ) in enumerate(θ))
                @test linear_combination(basis, θ, 𝑑(x))[1] ≈
                    sum(chebyshev_cos_deriv(y, i) * θ for (i,θ) in enumerate(θ))
            end

            # check grid
            g = @inferred collect(grid(basis))
            @test length(g) == N
            a, b = extrema(domain(basis))
            @test all(a .≤ g .≤ b)
            @test all(x -> isapprox(chebyshev_cos_deriv(transform_to(PM1(), transformation, x), N + 2),
                                    0, atol = 1e-13), g)
        end
    end
end

@testset "Chebysev adjusted basis" begin
    transformation = SemiInfRational(; endpoint = 3.0, scale = 7.0)
    for grid_kind in (Interior(), Endpoints())
        level0 = 3
        basis0 = univariate_basis(Chebyshev(), grid_kind, transformation, level0)
        θ0 = randn(dimension(basis0))
        for Δ in 1:4
            basis = adjust_basis(basis0, Δ)
            θ = adjust_coefficients(θ0, basis0, basis)
            @test basis.level == basis0.level + Δ
            for _ in 1:10
                x = rand_in_domain(basis0)
                @test linear_combination(basis0, θ0, x) ≈ linear_combination(basis, θ, x)
            end
        end
    end
    @test adjust_basis(basis0, -3) ≡ nothing
end

@testset "univariate derivatives" begin
    for (transformation, N) in ((BoundedLinear(lower = -2, upper = 3), 5),
                                (SemiInfRational(endpoint = 0.7, scale = 0.3), 1),
                                (InfRational(center = 0.4, scale = 0.9), 1))
        basis = univariate_basis(Chebyshev(), Interior(), transformation, 3)
        D = 𝑑^Val(N)
        f = linear_combination(basis, randn(dimension(basis)))
        for _ in 1:50
            x = rand_in_domain(basis)
            y = f(D(x))
            for i in 0:N
                @test y[i] ≈ DD(f, x, i) atol = 1e-6
            end
        end
    end
end

# @testset "endpoint continuity for derivatives" begin
#     N = 10
#     basis = Chebyshev(InteriorGrid(), N)

#     # NOTE here we are checking that in some sense, derivatives give the right limit at
#     # endpoints for transformations to ∞. We use the analytical derivatives for
#     # comparison, based on the chain rule.
#     x_pinf = 𝑑(Inf)
#     x_minf = 𝑑(-Inf)

#     @testset "SemiInfRational endpoints continuity" begin
#         trans = SemiInfRational(2.3, 0.7)

#         for i in 1:N
#             θ = e_i(basis ∘ trans, i)
#             y_pinf = @inferred linear_combination(basis ∘ trans, θ, x_pinf)
#             @test y_pinf[0] == 1
#             @test y_pinf[1] == 0
#             y_minf = @inferred linear_combination(basis ∘ trans, θ, x_minf)
#             @test y_minf[0] == 1
#             @test y_minf[1] == 0
#         end
#     end

#     @testset "InfRational endpoints continuity" begin
#         trans = InfRational(0.3, 3.0)

#         for i in 1:N
#             θ = e_i(basis ∘ trans, i)
#             y_pinf = @inferred linear_combination(basis ∘ trans, θ, x_pinf)
#             @test y_pinf[0] == 1
#             @test y_pinf[1] == 0
#             y_minf = @inferred linear_combination(basis ∘ trans, θ, x_minf)
#             @test y_minf[0] == (-1)^(i+1)
#             @test y_minf[1] == 0
#         end
#     end
# end
