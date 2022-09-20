using SpectralKit: nesting_total_length, nesting_block_length, SmolyakIndices,
    __smolyak_length, SmolyakGridShuffle, SmolyakProduct

"grids we test on"
GRIDS = (EndpointGrid(), InteriorGrid(), InteriorGrid2())

####
#### blocks
####

@testset "block length" begin
    for grid_kind in GRIDS
        for b in 0:5
            nA = nesting_total_length(Chebyshev, grid_kind, b)
            gA = grid(Chebyshev(grid_kind, nA))
            gB = grid(Chebyshev(grid_kind, nesting_total_length(Chebyshev, grid_kind, b + 1)))

            @test is_approximate_subset(gA, gB)
            @test sum(b -> nesting_block_length(Chebyshev, grid_kind, b), 0:b) == nA
        end
    end
end

"""
Collect shuffled indices for the given `grid_kind` for block indices `0, …, b`, returned as
a `Vector{Vector{Int}}`. For testing.
"""
function shuffled_indices_upto_b(grid_kind, b)
    _grid(b) = grid(Chebyshev(grid_kind, nesting_total_length(Chebyshev, grid_kind, b)))
    g0 = _grid(b)
    indices = Vector{Vector{Int}}()
    for b in b:(-1):1
        in_b = is_approximately_in(g0, _grid(b))
        notin_bm1 = .!is_approximately_in(g0, _grid(b - 1))
        mask = in_b .& notin_bm1
        push!(indices, findall(mask))
    end
    push!(indices, [(length(g0) + 1) ÷ 2])
    reverse(indices)
end

@testset "block shuffle" begin
    @testset "endpoint" begin
        for grid_kind in GRIDS
            for b in 0:6
                len = nesting_total_length(Chebyshev, grid_kind, b)
                ι = SmolyakGridShuffle(grid_kind, len)
                @test length(ι) == len
                @test @inferred eltype(ι) == Int
                @test collect(ι) == reduce(vcat, shuffled_indices_upto_b(grid_kind, b))
            end
        end
    end
end

####
#### traversal
####

"""
Naive implementation of Smolyan index iteration, traversing a `CartesianIndices` and keeping
valid indexes. For testing/comparison. Returns a vector of `indexes => blocks` pairs.
"""
function smolyak_indices_check(grid_kind, N, B, M)
    m = nesting_total_length(Chebyshev, grid_kind, M)
    b_table = fill(M, m)
    for b in (M-1):(-1):0
        b_table[1:nesting_total_length(Chebyshev, grid_kind, b)] .= b
    end
    T = NTuple{N,Int}
    result = Vector{Pair{T,T}}()
    for ι in CartesianIndices(ntuple(_ -> 1:m, N))
        ix = Tuple(ι)
        blocks = map(i -> b_table[i], ix)
        if sum(blocks) ≤ B
            push!(result, ix => blocks)
        end
    end
    result
end

@testset "Smolyak indices" begin
    for grid_kind in GRIDS
        for B in 0:3
            for M in 0:B
                for N in 1:4
                    ι = SmolyakIndices{N}(grid_kind, SmolyakParameters(B, M))
                    x1 = @inferred collect(ι)
                    x2 = first.(smolyak_indices_check(grid_kind, N, B, M))
                    len = @inferred __smolyak_length(grid_kind, Val(N), Val(B), M)
                    @test x1 == x2
                    @test len == length(x1) == length(ι) == length(x2)
                end
            end
        end
    end
end

@testset "Smolyak product primitives" begin
    for grid_kind in GRIDS
        for B in 0:3
            for M in 0:B
                for N in 1:4
                    ι = SmolyakIndices{N}(grid_kind, SmolyakParameters(B, M))
                    ℓ = nesting_total_length(Chebyshev, grid_kind, min(B,M))
                    sources = ntuple(_ -> rand(SVector{ℓ, Float64}), Val(N))
                    P = SmolyakProduct(ι, sources)
                    @test length(ι) == length(P)
                    @test eltype(P) == Float64
                    for (i, p) in zip(ι, P)
                        @test prod(getindex.(sources, i)) ≈ p
                    end
                end
            end
        end
    end
end

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
    @test @inferred(domain(basis)) == ((-1, 1), (-1, 1))
    g = grid(Float64, basis)
    iterator_sanity_checks(g)
    x = @inferred collect(g)
    M = @inferred collocation_matrix(basis, x)
    θ = M \ f.(x)
    @test sum(abs.(θ) .> 1e-8) == 4
    y1, y2 = range(domain(basis)[1]...; length = 100), range(domain(basis)[2]...; length = 100)
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
