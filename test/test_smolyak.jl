using SpectralKit: _chebyshev_open_ℓ, SmolyakIndices

@testset "Smolyak length" begin
    @test _chebyshev_open_ℓ(0) == 1
    @test _chebyshev_open_ℓ(1) == 1 + 2
    @test _chebyshev_open_ℓ(2) == 1 + 2 + 2
    @test _chebyshev_open_ℓ(3) == 1 + 2 + 2 + 4
    @test _chebyshev_open_ℓ(4) == 1 + 2 + 2 + 4 + 8
end

"""
Naive implementation of Smolyan index iteration, traversing a `CartesianIndices` and keeping
valid indexes. For testing/comparison.
"""
function smolyak_indices_check(N, ℓ, B, M)
    m = ℓ(M)
    b_table = fill(M, m)
    for b in (M-1):(-1):0
        b_table[1:ℓ(b)] .= b
    end
    result = Vector{NTuple{N,Int}}()
    for ι in CartesianIndices(ntuple(_ -> 1:m, N))
        ix = Tuple(ι)
        blocks = map(i -> b_table[i], ix)
        if sum(blocks) ≤ B
            push!(result, ix)
        end
    end
    result
end

@testset "Smolyak indices" begin
    for B in 0:3
        for M in 0:B
            for N in 1:4
                ι = SmolyakIndices{N}(_chebyshev_open_ℓ, B, M)
                x1 = collect(ι)
                x2 = smolyak_indices_check(N, _chebyshev_open_ℓ, B, M)
                @test x1 == x2
            end
        end
    end
end
