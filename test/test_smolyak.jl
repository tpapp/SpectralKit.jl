####
#### api
####

@testset "printing SmolyakLevels" begin
    @test repr(SmolyakLevel(; total = 3, each = 2)) == "Smolyak parameters, ∑ℓᵢ ≤ 3, all ℓᵢ ≤ 2"
end

@testset "Smolyak API checks" begin
    @test_logs (:warn, "‘each’ normalized to ‘total’") SmolyakLevel(total = 2, each = 4)
end

@testset "Smolyak API sanity checks" begin
    f(x) = (x[1] - 3) * (x[2] + 5) # linear function, just a sanity check
    transformations = (BoundedLinear(lower = 2, upper = 3), # approximation should be exact
                       BoundedLinear(lower = 3.0, upper = 4.5))
    basis = SmolyakBasis(Chebyshev(), Interior(), transformations, SmolyakLevel(total = 2))
    @test @inferred(domain(basis)) ≡ domain.(transformations)
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
            @test linear_combination(basis, θ, y) ≈ f(y) atol = 1e-14
        end
    end

    # FIXME re-enable once we have derivatives
    # @testset "sanity check for derivatives" begin
    #     # NOTE this just checks that it runs and is inferred, but does not check
    #     # correctness, derivatives derived below should be compared
    #     # x[1] * x[2] + 5 * x[1] - 3 * x[2] + 5
    #     # f1(x) = x[2] + 5
    #     # f2(x) = x[1] - 3
    #     # f12(x) = 1
    #     D = ∂(1, 1)
    #     y = SVector(1.0, 2.0)
    #     @test @inferred(linear_combination(basis, θ, D(y))) isa ∂Expansion
    # end
end

@testset "Smolyak API allocations" begin
    t = SemiInfRational()
    basis = SmolyakBasis(Chebyshev(), Interior(), (t, t),
                         SmolyakLevel(total = 3))
    y = SVector(0.4, 0.7)
    θ = randn(dimension(basis))
    @inferred linear_combination(basis, θ, y)
    @test @ballocated(linear_combination($basis, $θ, $y)) == 0
end

@testset "smolyak indices" begin
    for kind in KINDS
        for N in 1:5
            for total in 0:4
                for each in 0:total
                    expected = naive_smolyak_indices(Chebyshev(), kind, Val(N), total, each)
                    basis = SmolyakBasis(Chebyshev(), kind, ntuple(_ -> nothing, Val(N)),
                                         SmolyakLevel(; total, each))
                    @test collect(SpectralKit.SmolyakIndices(basis)) == expected
                end
            end
        end
    end
end


###
### augment coefficients
###

# @testset "Smolyak augment coefficients" begin
#     basis1 = SmolyakBasis(Chebyshev, InteriorGrid(), SmolyakParameters(2, 2), 2)
#     θ1 = randn(dimension(basis1))

#     # grid ≠
#     basis2_G = SmolyakBasis(Chebyshev, EndpointGrid(), SmolyakParameters(2, 3), 2)
#     @test !is_subset_basis(basis1, basis2_G)
#     @test_throws ArgumentError augment_coefficients(basis1, basis2_G, θ1)

#     # smolyak_parameters <
#     basis2_P = SmolyakBasis(Chebyshev, InteriorGrid(), SmolyakParameters(2, 1), 2)
#     @test !is_subset_basis(basis1, basis2_P)
#     @test_throws ArgumentError augment_coefficients(basis1, basis2_P, θ1)

#     # compatible basis
#     basis2 = SmolyakBasis(Chebyshev, InteriorGrid(), SmolyakParameters(3, 2), 2)
#     θ2 = @inferred augment_coefficients(basis1, basis2, θ1)
#     @test length(θ2) == dimension(basis2)
#     @test eltype(θ2) == eltype(θ1)
#     for _ in 1:100
#         x = (rand(), rand()) .* 4
#         @test linear_combination(basis1, θ1, x) ≈ linear_combination(basis2, θ2, x)
#     end
# end

# @testset "Smolyak nesting" begin
#     for grid_kind in GRIDS
#         for M1 in 0:5
#             for M2 in (M1 + 1):5
#                 for B1 in 0:M1
#                     for B2 in (B1 + 1):M2
#                         basis1 = SmolyakBasis(Chebyshev, grid_kind, SmolyakParameters(B1, M1), 2)
#                         basis2 = SmolyakBasis(Chebyshev, grid_kind, SmolyakParameters(B2, M2), 2)
#                         @test is_approximate_subset(collect(grid(basis1)), collect(grid(basis2)))
#                     end
#                 end
#             end
#         end
#     end
# end
