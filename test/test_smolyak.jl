using SpectralKit: block_length, cumulative_block_length, SmolyakIndices,
    smolyak_length, block_shuffle, __update_products, SmolyakProduct

####
#### blocks
####

@testset "block length" begin
    kind = InteriorGrid()
    @test cumulative_block_length(kind, 0) == 1
    @test cumulative_block_length(kind, 1) == 1 + 2
    @test cumulative_block_length(kind, 2) == 1 + 2 + 2
    @test cumulative_block_length(kind, 3) == 1 + 2 + 2 + 4
    @test cumulative_block_length(kind, 4) == 1 + 2 + 2 + 4 + 8
    for i in 1:10
        c = cumulative_block_length(kind, i)
        cprev = i == 0 ? 0 : cumulative_block_length(kind, i - 1)
        @test block_length(kind, i) == c - cprev
    end
end

@testset "block shuffle" begin
    results = Dict([0 => [1],
                    1 => [2, 1, 3],
                    2 => [3, 1, 5, 2, 4],
                    3 => vcat([5, 1, 9, 3, 7], 2:2:8),
                    4 => vcat([9, 1, 17, 5, 13], 3:4:17, 2:2:17)])
    kind = InteriorGrid()
    for (b, i) in pairs(results)
        len = length(i)
        @test len == cumulative_block_length(kind, b)
        ι = block_shuffle(kind, len)
        @test length(ι) == len
        @test @inferred eltype(ι) == Int
        @test collect(ι) == i
    end
end

####
#### traversal
####

"""
Naive implementation of Smolyan index iteration, traversing a `CartesianIndices` and keeping
valid indexes. For testing/comparison. Returns a vector of `indexes => blocks` pairs.
"""
function smolyak_indices_check(N, B, kind, M)
    m = cumulative_block_length(kind, M)
    b_table = fill(M, m)
    for b in (M-1):(-1):0
        b_table[1:cumulative_block_length(kind, b)] .= b
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
    kind = InteriorGrid()
    for B in 0:3
        for M in 0:B
            for N in 1:4
                ι = SmolyakIndices{N,B}(kind, M)
                x1 = @inferred collect(ι)
                x2 = first.(smolyak_indices_check(N, B, kind, M))
                len = @inferred smolyak_length(Val(N), Val(B), kind, M)
                @test x1 == x2
                @test len == length(x1) == length(ι) == length(x2)
            end
        end
    end
end

@testset "__update_products" begin
    N = 7
    M = 6
    sources = SVector{N}([rand(SVector{M,Float64}) for _ in 1:N])
    indices = SVector{N}([rand(1:M) for _ in 1:N])
    products = reverse(cumprod(reverse(getindex.(sources, indices))))
    @test products == @inferred __update_products(0, indices, sources, products)
    @test products == @inferred __update_products(1, indices, sources, zero_upto(products, 1))
    @test products == @inferred __update_products(2, indices, sources, zero_upto(products, 2))
    @test products == @inferred __update_products(N, indices, sources, zero_upto(products, N))
end

@testset "Smolyak product primitives" begin
    kind = InteriorGrid()
    for B in 0:3
        for M in 0:B
            for N in 1:4
                ι = SmolyakIndices{N,B}(kind, M)
                ℓ = cumulative_block_length(kind, min(B,M))
                sources = SVector{N}([rand(SVector{ℓ, Float64}) for _ in 1:N])
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

####
#### api
####

@testset "Smolyak API" begin
    basis = smolyak_basis(Chebyshev, InteriorGrid(),
                          Val(3), (BoundedLinear(0, 4), BoundedLinear(0, 3)))
    @test domain(basis) == ((0, 4), (0, 3))
    x = grid(Float64, basis, InteriorGrid())
    M = collocation_matrix(basis, x)
end
