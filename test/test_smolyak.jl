using SpectralKit: __block_length, __cumulative_block_length, SmolyakIndices,
    __smolyak_length, SmolyakGridShuffle, SmolyakProduct

####
#### blocks
####

@testset "block length" begin
    @test __cumulative_block_length(0) == 1
    @test __cumulative_block_length(1) == 1 + 2
    @test __cumulative_block_length(2) == 1 + 2 + 2
    @test __cumulative_block_length(3) == 1 + 2 + 2 + 4
    @test __cumulative_block_length(4) == 1 + 2 + 2 + 4 + 8
    for i in 1:10
        c = __cumulative_block_length(i)
        cprev = i == 0 ? 0 : __cumulative_block_length(i - 1)
        @test __block_length(i) == c - cprev
    end
end

@testset "block shuffle" begin
    results = Dict([0 => [1],
                    1 => [2, 1, 3],
                    2 => [3, 1, 5, 2, 4],
                    3 => vcat([5, 1, 9, 3, 7], 2:2:8),
                    4 => vcat([9, 1, 17, 5, 13], 3:4:17, 2:2:17)])
    for (b, i) in pairs(results)
        len = length(i)
        @test len == __cumulative_block_length(b)
        ι = SmolyakGridShuffle(len)
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
function smolyak_indices_check(N, B, M)
    m = __cumulative_block_length(M)
    b_table = fill(M, m)
    for b in (M-1):(-1):0
        b_table[1:__cumulative_block_length(b)] .= b
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
    for B in 0:3
        for M in 0:B
            for N in 1:4
                ι = SmolyakIndices{N}(SmolyakParameters{B}(M))
                x1 = @inferred collect(ι)
                x2 = first.(smolyak_indices_check(N, B, M))
                len = @inferred __smolyak_length(Val(N), Val(B), M)
                @test x1 == x2
                @test len == length(x1) == length(ι) == length(x2)
            end
        end
    end
end

@testset "Smolyak product primitives" begin
    for B in 0:3
        for M in 0:B
            for N in 1:4
                ι = SmolyakIndices{N}(SmolyakParameters{B}(M))
                ℓ = __cumulative_block_length(min(B,M))
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

@testset "Smolyak API sanity checks" begin
    f(x) = (x[1] - 3) * (x[2] + 5) # linear function, just a sanity check
    basis = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters{3}(3),
                          (BoundedLinear(0, 4), BoundedLinear(0, 3)))
    @test @inferred(domain(basis)) == ((0, 4), (0, 3))
    x = @inferred grid(Float64, basis)
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
    basis = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters{3}(3),
                          (BoundedLinear(0, 4), BoundedLinear(0, 3)))
    y = SVector(1.0, 2.0)
    θ = randn(dimension(basis))
    @inferred linear_combination(basis, θ, y)
    @test @ballocated(linear_combination($basis, $θ, $y)) == 0
end

@testset "Smolyak approximation" begin
    function f(x)               # cross-exp is particularly nasty for Smolyak
        x1, x2, x3, x4 = x
        exp(0.3 * (x1 - 2.0) * (x2 - 1.0)) + exp(-x3 * (x2 + 5)) - 1 / log(x4^2 + 10)
    end
    basis = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters{5}(5),
                          (BoundedLinear(0, 4), BoundedLinear(-2.0, 2.0),
                           SemiInfRational(1.0, 2.0), InfRational(0, 4)))
    x = grid(basis)
    θ = collocation_matrix(basis, x) \ f.(x)
    for x in x
        @test linear_combination(basis, θ, x) ≈ f(x) atol = 1e-10
    end
    s = SobolSeq([0, -2, 1, -10], [4, 2, 5, 10])
    skip(s, 1 << 5)
    Δabs, Δrel = mapreduce((x, y) -> max.(x, y), Iterators.take(s, 10^5)) do x
        fx = f(x)
        Δabs = abs(fx - linear_combination(basis, θ, x))
        Δrel = Δabs / (1 + abs(fx))
        Δabs, Δrel
    end
    @test Δabs ≤ 1e-3
    @test Δrel ≤ 5e-4
end

@testset "Smolyak derivatives" begin
    function f(x)
        x1, x2 = x
        7.0 + x1 + 2*x2 + 3*x2*x1
    end
    basis = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters{2}(2),
                          (BoundedLinear(0, 4), BoundedLinear(-5.0, 5.0)))
    x = grid(basis)
    θ = collocation_matrix(basis, x) \ f.(x)
    F = linear_combination(basis, θ)
    for x in x
        @test F(x) ≈ f(x) atol = 1e-10
        @test gradient(F, x) ≈ gradient(f, x)
    end
end
