var documenterSearchIndex = {"docs":
[{"location":"#SpectralKit","page":"SpectralKit","title":"SpectralKit","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"This is a very simple package for building blocks of spectral methods. Its intended audience is users who are familiar with the theory and practice of these methods, and prefer to assemble their code from modular building blocks. If you need an introduction, a book like Boyd (2001): Chebyshev and Fourier spectral methods is a good place to start.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"The package is optimized for solving functional equations, as usually encountered in economics when solving discrete and continuous-time problems. It uses static arrays extensively to avoid allocation and unroll some loops. Key functionality includes evaluating a set of basis functions, their linear combination at arbitrary points in a fast manner, for use in threaded code. These should work seamlessly with automatic differentiation frameworks when derivatives are needed.","category":"page"},{"location":"#Introduction","page":"SpectralKit","title":"Introduction","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"In this package,","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"A basis is a finite family of functions for approximating other functions. The dimension of a basis tells you how many functions are in there, while domain can be used to query its domain.\nA grid is vector of suggested gridpoints for evaluating the function to be approximated that has useful theoretical properties. You can contruct a collocation_matrix using this grid (or any other set of points). Grids are associated with bases at the time of their construction: a basis with the same set of functions can have different grids.\nbasis_at returns an iterator for evaluating basis functions at an arbitrary point inside their domain. This iterator is meant to be heavily optimized and non-allocating. linear_combination is a convenience wrapper for obtaining a linear combination of basis functions at a given point.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"A basis is constructed using","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"a family on a fixed domain, eg Chebyshev,\na grid specification like InteriorGrid,\na number of parameters that determine the number of basis functions.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"A set of coordinates for a particular basis can be augmented for a wider basis with augment_coefficients.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"Currenly, all bases have the domain -11 or -11^n. Facilities are provided for coordinatewise transformations to other domains.","category":"page"},{"location":"#Examples","page":"SpectralKit","title":"Examples","text":"","category":"section"},{"location":"#Univariate-family-on-[-1,1]","page":"SpectralKit","title":"Univariate family on [-1,1]","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"using SpectralKit\nbasis = Chebyshev(EndpointGrid(), 5)   # 5 Chebyshev polynomials\nis_function_basis(basis)               # ie we support the interface below\ndimension(basis)                       # number of basis functions\ndomain(basis)                          # domain\ngrid(basis)                            # Gauss-Lobatto grid\ncollect(basis_at(basis, 0.41))         # iterator for basis functions at 0.41\nθ = [1, 0.5, 0.2, 0.3, 0.001]          # a vector of coefficients\nlinear_combination(basis, θ, 0.41)     # combination at some value\nlinear_combination(basis, θ)(0.41)     # also as a callable\nbasis2 = Chebyshev(EndpointGrid(), 8)  # 8 Chebyshev polynomials\nis_subset_basis(basis, basis2)         # we could augment θ …\naugment_coefficients(basis, basis2, θ) # … so let's do it","category":"page"},{"location":"#Smolyak-approximation-on-a-transformed-domain","page":"SpectralKit","title":"Smolyak approximation on a transformed domain","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"using SpectralKit, StaticArrays\nfunction f2(x)                  # bivariate function we approximate\n    x1, x2 = x                  # takes vectors\n    exp(x1) + exp(-abs2(x2))\nend\nct = coordinate_transformations(BoundedLinear(-1, 2.0), SemiInfRational(-3.0, 3.0))\nbasis = smolyak_basis(Chebyshev, InteriorGrid2(), SmolyakParameters(3), 2)\nx = grid(basis)\nθ = collocation_matrix(basis) \\ f2.(from_pm1.(ct, x)) # find the coefficients\nz = SVector(0.5, 0.7)                                 # evaluate at this point\nisapprox(f2(z), linear_combination(basis, θ, to_pm1(ct, z)), rtol = 0.005)","category":"page"},{"location":"#Constructing-bases","page":"SpectralKit","title":"Constructing bases","text":"","category":"section"},{"location":"#Grid-specifications","page":"SpectralKit","title":"Grid specifications","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"EndpointGrid\nInteriorGrid\nInteriorGrid2","category":"page"},{"location":"#SpectralKit.EndpointGrid","page":"SpectralKit","title":"SpectralKit.EndpointGrid","text":"struct EndpointGrid <: SpectralKit.AbstractGrid\n\nGrid that includes endpoints (eg Gauss-Lobatto).\n\nnote: Note\nFor small dimensions may fall back to a grid that does not contain endpoints.\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.InteriorGrid","page":"SpectralKit","title":"SpectralKit.InteriorGrid","text":"struct InteriorGrid <: SpectralKit.AbstractGrid\n\nGrid with interior points (eg Gauss-Chebyshev).\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.InteriorGrid2","page":"SpectralKit","title":"SpectralKit.InteriorGrid2","text":"struct InteriorGrid2 <: SpectralKit.AbstractGrid\n\nGrid with interior points that results in smaller grids than InteriorGrid when nested. Equivalent to an EndpointGrid with endpoints dropped.\n\n\n\n\n\n","category":"type"},{"location":"#Univariate-and-multivariate-transformations","page":"SpectralKit","title":"Univariate and multivariate transformations","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"Bases are defined on the domain -1 1 or -1 1^n. Transformations map other uni- and multivariate sets into these domains.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"to_pm1\nfrom_pm1\nBoundedLinear\nInfRational\nSemiInfRational\ncoordinate_transformations","category":"page"},{"location":"#SpectralKit.to_pm1","page":"SpectralKit","title":"SpectralKit.to_pm1","text":"to_pm1(transformation, x)\n\nTransform x to -1 1 using transformation.\n\nSupports partial application as to_pm1(transformation).\n\n!!! FIXME     document, especially differentiability requirements at infinite endpoints\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.from_pm1","page":"SpectralKit","title":"SpectralKit.from_pm1","text":"from_pm1(transformation, x)\n\nTransform x from -1 1 using transformation.\n\nSupports partial application as from_pm1(transformation).\n\n!!! FIXME     document, especially differentiability requirements at infinite endpoints\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.BoundedLinear","page":"SpectralKit","title":"SpectralKit.BoundedLinear","text":"struct BoundedLinear{T<:Real} <: SpectralKit.UnivariateTransformation\n\nTransform x ∈ (-1,1) to y ∈ (a, b), using y = x  s + m.\n\nm and s are calculated and checked by the constructor; a < b is enforced.\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.InfRational","page":"SpectralKit","title":"SpectralKit.InfRational","text":"InfRational(A, L)\n\n\nChebyshev polynomials transformed to the domain (-Inf, Inf) using y = A + L  x  (1 - x^2), with L > 0.\n\nExample mappings\n\n0  A\n05  A  L  3\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.SemiInfRational","page":"SpectralKit","title":"SpectralKit.SemiInfRational","text":"SemiInfRational(A, L)\n\n\n[-1,1] transformed to the domain [A, Inf) (when L > 0) or (-Inf,A] (when L < 0) using y = A + L  (1 + x)  (1 - x).\n\nWhen used with Chebyshev polynomials, also known as a “rational Chebyshev” basis.\n\nExample mappings\n\n-12  A + L  3\n0  A + L\n12  A + 3  L\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.coordinate_transformations","page":"SpectralKit","title":"SpectralKit.coordinate_transformations","text":"coordinate_transformations(transformations)\n\n\nWrapper for coordinate-wise transformations.\n\njulia> using StaticArrays\n\njulia> ct = coordinate_transformations(BoundedLinear(0, 2), SemiInfRational(2, 3))\ncoordinate transformations\n  (0.0,2.0) ↔ (-1, 1) [linear transformation]\n  (2,∞) ↔ (-1, 1) [rational transformation with scale 3]\n\njulia> x = from_pm1(ct, SVector(0.4, 0.5))\n2-element SVector{2, Float64} with indices SOneTo(2):\n  1.4\n 11.0\n\njulia> y = to_pm1(ct, x)\n2-element SVector{2, Float64} with indices SOneTo(2):\n 0.3999999999999999\n 0.5\n\n\n\n\n\n","category":"function"},{"location":"#Univariate-bases","page":"SpectralKit","title":"Univariate bases","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"Currently, only Chebyshev polynomials are implemented. Univariate bases operate on real numbers.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"Chebyshev","category":"page"},{"location":"#SpectralKit.Chebyshev","page":"SpectralKit","title":"SpectralKit.Chebyshev","text":"struct Chebyshev{K} <: SpectralKit.FunctionBasis\n\nThe first N Chebyhev polynomials of the first kind, defined on [-1,1].\n\n\n\n\n\n","category":"type"},{"location":"#Multivariate-bases","page":"SpectralKit","title":"Multivariate bases","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"Multivariate bases operate on vectors. StaticArrays.SVector is preferred for performance, but all <:AbstractVector types should work.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"SmolyakParameters\nsmolyak_basis","category":"page"},{"location":"#SpectralKit.SmolyakParameters","page":"SpectralKit","title":"SpectralKit.SmolyakParameters","text":"SmolyakParameters(B)\nSmolyakParameters(B, M)\n\n\nParameters for Smolyak grids that are independent of the dimension of the domain.\n\nPolynomials are organized into blocks of 1, 2, 2, 4, 8, 16, … polynomials (and corresponding gridpoints), indexed with a block index b that starts at 0. B ≥ ∑ bᵢ and 0 ≤ bᵢ ≤ M constrain the number of blocks along each dimension i.\n\nM > B is not an error, but will be normalized to M = B.\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.smolyak_basis","page":"SpectralKit","title":"SpectralKit.smolyak_basis","text":"smolyak_basis(univariate_family, grid_kind, smolyak_parameters, _)\n\n\nCreate a sparse Smolyak basis.\n\nArguments\n\nunivariate_family: should be a callable that takes a grid_kind and a dimension parameter, eg Chebyshev.\ngrid_kind: the grid kind, eg InteriorGrid() etc.\nsmolyak_parameters: the Smolyak grid specification parameters, see SmolyakParameters.\nN: the dimension. wrapped in a Val for type stability, a convenience constructor also takes integers.\n\nExample\n\njulia> basis = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(3), 2)\nSparse multivariate basis on ℝ^2\n  Smolyak indexing, ∑bᵢ ≤ 3, all bᵢ ≤ 3, dimension 81\n  using Chebyshev polynomials (1st kind), InteriorGrid(), dimension: 27\n\njulia> dimension(basis)\n81\n\njulia> domain(basis)\n((-1, 1), (-1, 1))\n\nProperties\n\nGrids nest: increasing arguments of SmolyakParameters result in a refined grid that contains points of the cruder grid.\n\n\n\n\n\n","category":"function"},{"location":"#Using-bases","page":"SpectralKit","title":"Using bases","text":"","category":"section"},{"location":"#Introspection","page":"SpectralKit","title":"Introspection","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"is_function_basis\ndimension\ndomain","category":"page"},{"location":"#SpectralKit.is_function_basis","page":"SpectralKit","title":"SpectralKit.is_function_basis","text":"is_function_basis(F)\n\nis_function_basis(f::F)\n\nTest if the argument is a function basis, supporting the following interface:\n\ndomain for querying the domain,\ndimension for the dimension,\nbasis_at for function evaluation,\ngrid to obtain collocation points.\n\nlinear_combination and collocation_matrix are also supported, building on the above.\n\nCan be used on both types (preferred) and values (for convenience).\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.dimension","page":"SpectralKit","title":"SpectralKit.dimension","text":"dimension(basis)\n\nReturn the dimension of basis, a positive Int.\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.domain","page":"SpectralKit","title":"SpectralKit.domain","text":"domain(basis)\n\nThe domain of a function basis. A tuple of numbers (of arbitrary type, but usually Float64), or a tuple of domains by coordinate.\n\n\n\n\n\n","category":"function"},{"location":"#Evaluation","page":"SpectralKit","title":"Evaluation","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"basis_at\nlinear_combination","category":"page"},{"location":"#SpectralKit.basis_at","page":"SpectralKit","title":"SpectralKit.basis_at","text":"basis_at(basis, x)\n\nReturn an iterable with known element type and length (Base.HasEltype(), Base.HasLength()) of basis functions in basis evaluated at x.\n\nUnivariate bases operate on real numbers, while for multivariate bases, StaticArrays.SVector is preferred for performance, though all <:AbstractVector types should work.\n\nMethods are type stable.\n\nnote: Note\nConsequences are undefined when evaluating outside the domain.\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.linear_combination","page":"SpectralKit","title":"SpectralKit.linear_combination","text":"linear_combination(basis, θ, x)\n\n\nEvaluate the linear combination of  θₖfₖ(x) of function basis f₁  at x, for the given order.\n\nThe length of θ should equal dimension(θ).\n\n\n\n\n\nlinear_combination(basis, θ)\n\n\nReturn a callable that calculates linear_combination(basis, θ, x) when called with x.\n\n\n\n\n\n","category":"function"},{"location":"#Grids-and-collocation","page":"SpectralKit","title":"Grids and collocation","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"grid\ncollocation_matrix","category":"page"},{"location":"#SpectralKit.grid","page":"SpectralKit","title":"SpectralKit.grid","text":"grid([T], basis)\n\nReturn a grid recommended for collocation, with dimension(basis) elements.\n\nT is used as a hint for the element type of grid coordinates, and defaults to Float64. The actual type can be broadened as required. Methods are type stable.\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.collocation_matrix","page":"SpectralKit","title":"SpectralKit.collocation_matrix","text":"collocation_matrix(basis)\ncollocation_matrix(basis, x)\n\n\nConvenience function to obtain a collocation matrix at gridpoints x, which is assumed to have a concrete eltype. The default is x = grid(basis), specialized methods may exist for this when it makes sense.\n\nMethods are type stable.\n\n\n\n\n\n","category":"function"},{"location":"#Augment-coordinates-for-a-wider-basis","page":"SpectralKit","title":"Augment coordinates for a wider basis","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"augment_coefficients\nis_subset_basis","category":"page"},{"location":"#SpectralKit.augment_coefficients","page":"SpectralKit","title":"SpectralKit.augment_coefficients","text":"augment_coefficients(basis1, basis2, θ1)\n\nReturn a set of coefficients θ2 for basis2 such that\n\nlinear_combination(basis1, θ1, x) == linear_combination(basis2, θ2, x)\n\nfor any x in the domain. In practice this means padding with zeros.\n\nThrow a ArgumentError if the bases are incompatible with each other or x, or this is not possible. Methods may not be defined for incompatible bases, compatibility between bases can be checked with is_subset_basis.\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.is_subset_basis","page":"SpectralKit","title":"SpectralKit.is_subset_basis","text":"is_subset_basis(basis1, basis2)\n\n\nReturn a Bool indicating whether coefficients in basis1 can be augmented to basis2 with augment_coefficients.\n\nnote: Note\ntrue does not mean that coefficients from basis1 can just be padded with zeros, since they may be in different positions. Always use augment_coefficients.\n\n\n\n\n\n","category":"function"},{"location":"#Internals","page":"SpectralKit","title":"Internals","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"This section of the documentation is probably only relevant to contributors and others who want to understand the internals.","category":"page"},{"location":"#Simplified-API-for-adding-custom-transformations","page":"SpectralKit","title":"Simplified API for adding custom transformations","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"SpectralKit.UnivariateTransformation","category":"page"},{"location":"#SpectralKit.UnivariateTransformation","page":"SpectralKit","title":"SpectralKit.UnivariateTransformation","text":"abstract type UnivariateTransformation\n\nAn abstract type for univariate transformations. Transformations are not required to be subtypes, this just documents the interface they need to support:\n\nto_pm1\nfrom_pm1\ndomain\n\n!!! NOTE     Abstract type used for code organization, not exported.\n\n\n\n\n\n","category":"type"},{"location":"#Grid-internals","page":"SpectralKit","title":"Grid internals","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"SpectralKit.gridpoint","category":"page"},{"location":"#SpectralKit.gridpoint","page":"SpectralKit","title":"SpectralKit.gridpoint","text":"gridpoint(basis, i)\n\n\nReturn a gridpoint for collocation, with 1 ≤ i ≤ dimension(basis).\n\nT is used as a hint for the element type of grid coordinates, and defaults to Float64. The actual type can be broadened as required. Methods are type stable.\n\nnote: Note\nNot all grids have this method defined, especially if it is impractical. See grid, which is part of the API, this function isn't.\n\n\n\n\n\n","category":"function"}]
}
