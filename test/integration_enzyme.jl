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
    autodiff(Reverse, g, Const(basis), Duplicated(x, dx))
end
