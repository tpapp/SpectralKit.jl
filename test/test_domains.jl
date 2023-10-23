@testset "domain API" begin
    d1 = SpectralKit.PM1()
    @test extrema(d1) == (-1, 1)
    @test domain_kind(d1) ≡ :univariate
    dn = coordinate_domains(d1, d1)
    @test domain_kind(dn) ≡ :multivariate
    @test length(dn) == 2
    @test dn[1] == d1
    @test Tuple(dn) ≡ (d1, d1)
end
