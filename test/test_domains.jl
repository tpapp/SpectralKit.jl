@testset "domain API" begin
    d1 = SpectralKit.PM1()
    @test repr(d1) == "[-1,1]"
    @test extrema(d1) == (-1, 1)
    @test (minimum(d1), maximum(d1)) == extrema(d1)
    @test domain_kind(d1) ≡ :univariate
    dn = coordinate_domains(d1, d1)
    @test dn == coordinate_domains((d1, d1)) == coordinate_domains(Val(2), d1)
    @test repr(dn) == "[-1,1]²"
    @test domain_kind(dn) ≡ :multivariate
    @test length(dn) == 2
    @test dn[1] == d1
    @test Tuple(dn) ≡ (d1, d1)
    d2 = coordinate_domains(SpectralKit.PM1(), SpectralKit.UnivariateDomain(-3.0, Inf))
    @test repr(d2) == "[-1,1]×[-3.0,∞]"
end
