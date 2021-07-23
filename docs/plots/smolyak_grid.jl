using SpectralKit, PGFPlotsX

basis = smolyak_basis(Chebyshev, EndpointGrid(), Val(3),
                      let b = BoundedLinear(-1,1); (b, b); end)
g = grid(basis)
p = @pgf Axis({scale_only_axis = true,
               width = "6cm",
               height = "6cm" },
              Plot({ only_marks, red }, Coordinates(Tuple.(g))))
pgfsave("smolyak_grid.png", p; dpi = 600)
