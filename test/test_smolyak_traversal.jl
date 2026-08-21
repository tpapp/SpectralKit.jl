@testset "smolyak step" begin
    XS = (2, 3, 5, 7, 11)
    for kind in KINDS
        for N in 1:5
            for total in 1:4
                for each in 1:4
                    xs = XS[1:N]
                    test_smolyak_step(Chebyshev(), Interior(), total, each,
                                      (a, b) -> isempty(b) ? a : a * first(b),
                                      map(SpectralKit.ChebyshevIterator, xs))
                end
            end
        end
    end
end

@testset "__smolyak_length" begin
    for kind in KINDS
        for N in 1:5
            for total in 1:4
                for each in 1:4
                    @test SpectralKit.__smolyak_length(Chebyshev(), kind, N, total, each) ==
                        length(naive_smolyak_indices(Chebyshev(), kind, Val(N), total, each))
                end
            end
        end
    end
end
