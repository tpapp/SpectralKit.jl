using PGFPlotsX, SpectralKit, Colors

function plot_family(family, ns, order, title;
                     xmin = domain_extrema(family)[1], xmax = domain_extrema(family)[2],
                     rel_offsets = [0.01, 0.01])
    axis = @pgf Axis({ xmin = xmin, xmax = xmax, xlabel = "y", title = title,
                       legend_pos = "outer north east"})
    xgrid = range(xmin, xmax; length = 400)
    mi, ma = domain_extrema(family)
    colors = distinguishable_colors(length(ns); lchoices = 0:60)
    offsets = [1, -1] .* rel_offsets .* (xmax - xmin)
    for (n, col) in zip(ns, colors)
        y = evaluate.(family, n, xgrid, order)
        @pgf push!(axis,
                   Plot({ no_marks, thick, color = col }, Table(xgrid, y)),
                   LegendEntry(string(n)))
    end
    ex = [mi, ma]
    for (i, (n, col)) in enumerate(zip(ns, colors))
        y = evaluate.(family, n, ex, order)
        @pgf push!(axis, Plot({ only_marks, color = col },
                              Table([xmin, xmax] .+ offsets .* i, y)))
    end
    axis
end

p = plot_family(Chebyshev(), 0:5, Val(0), "Chebyshev polynomials (value)")
pgfsave("chebyshev.png", p; dpi = 600)

p = plot_family(Chebyshev(), 0:5, Val(1), "Chebyshev polynomials (derivative)";
                rel_offsets = [0, 0])
pgfsave("chebyshev_deriv.png", p; dpi = 600)

F = SemiInfChebyshev(0.0, 1.0)
p = plot_family(F, 0:5, Val(0), raw"Rational Chebyshev functions $[0,\infty)$ (value)";
                xmax = 1)
pgfsave("semiinf.png", p; dpi = 600)

p = plot_family(F, 0:5, Val(0), raw"Rational Chebyshev functions $[0,\infty)$ (value)";
                xmax = 200)
pgfsave("semiinf_birdseye.png", p; dpi = 600)

p = plot_family(F, 0:5, Val(1), raw"Rational Chebyshev functions $[0,\infty)$ (derivative)";
                xmax = 1, rel_offsets = [0, 0.01])
pgfsave("semiinf_deriv.png", p; dpi = 600)
