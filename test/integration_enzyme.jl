#####
##### integration test for Enzyme
#####

using Enzyme

@testset "Enzyme integration" begin
    basis = SmolyakBasis(Chebyshev(), Interior(),
                         ntuple(_ -> BoundedLinear(1.0, 2.0), Val(2)),
                         SmolyakLevel(total = 2, each = 2))
    g(basis, θ) = linear_combination(basis, θ, (1.5, 1.5))
    d = dimension(basis)
    x = zeros(d)
    dx = zeros(d)
    @test @inferred(autodiff(Reverse, g, Const(basis), Duplicated(x, dx))) == ((nothing, nothing),)
end

@testset "z over hypot torture test" begin
    V = [2.0, 3.0, 1e-180, 1e-180, 0.5e-100, 1e-100, 2e-100, 1e-50, 1e100, 1e200]
    for s in V
        for z in V
            for dz in V
                for ds in V
                    dx_z, x = Enzyme.autodiff(Enzyme.ForwardWithPrimal, SK._inf_to_pm1,
                                              Enzyme.Duplicated(z, dz), Enzyme.Const(s))
                    dx_s, = Enzyme.autodiff(Enzyme.Forward, SK._inf_to_pm1,
                                            Enzyme.Const(z), Enzyme.Duplicated(s, ds))
                    _z = BigFloat(z)
                    _s = BigFloat(s)
                    _r = _z / _s
                    _q = one(_r) + _r^2
                    _x = _z / √(_z^2 + _s^2)
                    _dx_dz = 1 / _s / (_q * √_q)
                    _dx_ds = -_dx_dz * _r
                    @test x ≈ Float64(_x)
                    @test dx_z ≈ Float64(_dx_dz * BigFloat(dz)) atol = 100 * eps() rtol = 100 * eps()
                    @test dx_s ≈ Float64(_dx_ds * BigFloat(ds)) atol = 100 * eps() rtol = 100 * eps()
                    (R_dx_dz, R_dx_ds), x = Enzyme.autodiff(Enzyme.ReverseWithPrimal,
                                                            SK._inf_to_pm1, Active,
                                                            Active(z), Active(s))
                    @test Float64(_dx_dz) ≈ R_dx_dz
                    @test Float64(_dx_ds) ≈ R_dx_ds
                end
            end
        end
    end
end
