using PGFPlotsX, SpectralKit, Colors, ForwardDiff

"""
Plot selected basis functions from a basis.

# Arguments

- `basis`: the basis we use
- `ns`: the polynomials which are picked
- `order`: 0 (value) or 1 (derivative)
- `title`: plot title

# Keyword arguments

- `xmin`, `xmax`: boundaries of the grid for calculating points
- `len`: number of points
- `rel_offsets`: extends the plotting area by a fraction (one value for each side)
"""
function plot_basis(basis, ns, order, title;
                    xmin = domain(basis)[1], xmax = domain(basis)[2],
                    rel_offsets = [0.01, 0.01], len = 500)
    @assert order ∈ 0:1
    axis = @pgf Axis({ xmin = xmin, xmax = xmax, xlabel = "y", title = title,
                       legend_pos = "outer north east"})
    xgrid = range(xmin, xmax; length = len)
    mi, ma = domain(basis)
    colors = distinguishable_colors(length(ns); lchoices = 0:60)
    offsets = [1, -1] .* rel_offsets .* (xmax - xmin)
    ev0(n, x) = first(Iterators.drop(basis_at(basis, x), n - 1)) # NOTE horribly inefficient …
    ev1(n, x) = ForwardDiff.derivative(x -> ev0(n, x), x) # NOTE … but does not matter
    ev(n, x) = order == 0 ? ev0(n, x) : ev1(n, x)
    for (n, col) in zip(ns, colors)
        y = map(x -> ev(n, x), xgrid)
        @pgf push!(axis,
                   Plot({ no_marks, thick, color = col }, Table(xgrid, y)),
                   LegendEntry(string(n)))
    end
    for (i, (n, col)) in enumerate(zip(ns, colors))
        y = [ev(n, mi), ev(n, ma)]
        @pgf push!(axis, Plot({ only_marks, color = col },
                              Table([xmin, xmax] .+ offsets .* i, y)))
    end
    axis
end

N = 5
ns = 1:N                        # polynomials we generate

###
### Chebyshev
###

p = plot_basis(Chebyshev(InteriorGrid(), N), ns, 0, "Chebyshev polynomials (value)")
pgfsave("chebyshev.png", p; dpi = 600)

p = plot_basis(Chebyshev(InteriorGrid(), N), ns, 1, "Chebyshev polynomials (derivative)";
                rel_offsets = [0, 0])
pgfsave("chebyshev_deriv.png", p; dpi = 600)

###
### Rational Chebysev [0,∞)
###

basis = univariate_basis(Chebyshev, InteriorGrid(), N, SemiInfRational(0.0, 1.0))
p = plot_basis(basis, ns, 0, raw"Rational Chebyshev functions $[0,\infty)$ (value)";
               xmax = 1)
pgfsave("semiinf.png", p; dpi = 600)

p = plot_basis(basis, ns, 0, raw"Rational Chebyshev functions $[0,\infty)$ (value)";
               xmax = 200)
pgfsave("semiinf_birdseye.png", p; dpi = 600)

p = plot_basis(basis, ns, 1, raw"Rational Chebyshev functions $[0,\infty)$ (derivative)";
               xmax = 1, rel_offsets = [0, 0.01])
pgfsave("semiinf_deriv.png", p; dpi = 600)

###
### Rational Chebysev (-∞,∞)
###

basis = univariate_basis(Chebyshev, InteriorGrid(), N, InfRational(0.0, 1.0))
p = plot_basis(basis, ns, 0,
               raw"Rational Chebyshev functions $(-\infty,\infty)$ (value)";
               xmin = -1, xmax = 1)
pgfsave("inf.png", p; dpi = 600)

p = plot_basis(basis, ns, 0,
               raw"Rational Chebyshev functions $(-\infty,\infty)$ (value)";
               xmin = -30, xmax = 30, len = 1000)
pgfsave("inf_birdseye.png", p; dpi = 600)

p = plot_basis(basis, ns, 1,
               raw"Rational Chebyshev functions $(-\infty,\infty)$ (derivative)";
               xmin = -5, xmax = 5, rel_offsets = [0.01, 0.01])
pgfsave("inf_deriv.png", p; dpi = 600)
