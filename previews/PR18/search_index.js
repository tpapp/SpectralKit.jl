var documenterSearchIndex = {"docs":
[{"location":"#SpectralKit","page":"SpectralKit","title":"SpectralKit","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"This is a very simple package for building blocks of spectral methods. Its intended audience is users who are familiar with the theory and practice of these methods, and prefer to assemble their code from modular building blocks. If you need an introduction, a book like Boyd (2001): Chebyshev and Fourier spectral methods is a good place to start.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"The package is optimized for solving functional equations, as usually encountered in economics when solving discrete and continuous-time problems. It uses static arrays extensively to avoid allocation and unroll some loops. Key functionality includes evaluating a set of basis functions, their linear combination at arbitrary points in a fast manner, for use in threaded code. These should work seamlessly with automatic differentiation frameworks when derivatives are needed.","category":"page"},{"location":"#Introduction","page":"SpectralKit","title":"Introduction","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"In this package,","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"A basis is a finite family of functions for approximating other functions. The dimension of a basis tells you how many functions are in there, while domain can be used to query its domain.\nA grid is vector of suggested gridpoints for evaluating the function to be approximated that has useful theoretical properties. You can contruct a collocation_matrix using this grid. Grids are associated with bases at the time of their construction: a basis with the same set of functions can have different grids.\nbasis_at returns an iterator for evaluating basis functions at an arbitrary point inside their domain. This iterator is meant to be heavily optimized and non-allocating. linear_combination is a convenience wrapper for obtaining a linear combination of basis functions at a given point.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"A basis is constructed using","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"a primitive family on a fixed domain, eg Chebyshev,\na grid specification like InteriorGrid,\na number of parameters that determine the number of basis functions.\ntransformation(s) that project the domain of a primitive family into the domain.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"The following example provides an overview of the interface for univariate function families:","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"using SpectralKit\nbasis = univariate_basis(Chebyshev, EndpointGrid(), # 5 Chebyshev polynomials on [0,2]\n                         5, BoundedLinear(0, 2))\nis_function_basis(basis) # tells us that we support the interface below\ndimension(basis) # number of basis functions\ndomain(basis)            # endpoints \ngrid(basis) # Gauss-Lobatto grid\ncollect(basis_at(basis, 0.41)) # iterator for basis functions at 0.41\nθ = [1, 0.5, 0.2, 0.3, 0.001] # a vector of coefficients\nlinear_combination(basis, θ, 0.41) # combination at some value\nlinear_combination(basis, θ)(0.41) # also as a callable","category":"page"},{"location":"#Constructing-bases","page":"SpectralKit","title":"Constructing bases","text":"","category":"section"},{"location":"#Primitive-families","page":"SpectralKit","title":"Primitive families","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"Currently, only Chebyshev polynomials are implemented.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"Chebyshev","category":"page"},{"location":"#SpectralKit.Chebyshev","page":"SpectralKit","title":"SpectralKit.Chebyshev","text":"struct Chebyshev{K} <: SpectralKit.FunctionBasis\n\nThe first N Chebyhev polynomials of the first kind, defined on [-1,1].\n\n\n\n\n\n","category":"type"},{"location":"#Grid-specifications","page":"SpectralKit","title":"Grid specifications","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"InteriorGrid\nEndpointGrid","category":"page"},{"location":"#SpectralKit.InteriorGrid","page":"SpectralKit","title":"SpectralKit.InteriorGrid","text":"struct InteriorGrid <: SpectralKit.AbstractGrid\n\nGrid with interior points (eg Gauss-Chebyshev).\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.EndpointGrid","page":"SpectralKit","title":"SpectralKit.EndpointGrid","text":"struct EndpointGrid <: SpectralKit.AbstractGrid\n\nGrid that includes endpoints (eg Gauss-Lobatto).\n\n\n\n\n\n","category":"type"},{"location":"#Transformations","page":"SpectralKit","title":"Transformations","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"BoundedLinear\nInfRational\nSemiInfRational","category":"page"},{"location":"#SpectralKit.BoundedLinear","page":"SpectralKit","title":"SpectralKit.BoundedLinear","text":"struct BoundedLinear{T<:Real} <: SpectralKit.UnivariateTransformation\n\nTransform x ∈ (-1,1) to y ∈ (a, b), using y = x  s + m.\n\nm and s are calculated and checked by the constructor; a < b is enforced.\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.InfRational","page":"SpectralKit","title":"SpectralKit.InfRational","text":"InfRational(A, L)\n\n\nChebyshev polynomials transformed to the domain (-Inf, Inf) using y = A + L  x  (1 - x^2), with L > 0.\n\n0 is mapped to A.\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.SemiInfRational","page":"SpectralKit","title":"SpectralKit.SemiInfRational","text":"SemiInfRational(A, L)\n\n\n[-1,1] transformed to the domain [A, Inf) (when L > 0) or (-Inf,A] (when L < 0) using y = A + L  (1 + x)  (1 - x).\n\nWhen used with Chebyshev polynomials, also known as a “rational Chebyshev” basis.\n\n\n\n\n\n","category":"type"},{"location":"#Univariate-bases","page":"SpectralKit","title":"Univariate bases","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"univariate_basis","category":"page"},{"location":"#SpectralKit.univariate_basis","page":"SpectralKit","title":"SpectralKit.univariate_basis","text":"univariate_basis(univariate_family, grid_kind, N, transformation)\n\n\nCreate a univariate basis from univariate_family, with the specified grid_kind and dimension N, transforming the domain with transformation.\n\nparent is a univariate basis, transformation is a univariate transformation (supporting the interface described by UnivariateTransformation, but not necessarily a subtype). Univariate bases support SpectralKit.gridpoint.\n\nExample\n\nThe following is a basis with 10 transformed Chebyshev polynomials of the first kind on (3), with equal amounts of nodes on both sides of 7 = 3 + 4 and an interior grid:\n\njulia> basis = univariate_basis(Chebyshev, InteriorGrid(), 10, SemiInfRational(3.0, 4.0))\nChebyshev polynomials (1st kind), interior grid, dimension: 10\n  on (3.0,∞) [rational transformation with scale 4.0]\n\njulia> dimension(basis)\n10\n\njulia> domain(basis)\n(3.0, Inf)\n\n\n\n\n\n","category":"function"},{"location":"#Multivariate-bases","page":"SpectralKit","title":"Multivariate bases","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"smolyak_basis","category":"page"},{"location":"#SpectralKit.smolyak_basis","page":"SpectralKit","title":"SpectralKit.smolyak_basis","text":"smolyak_basis(univariate_family, grid_kind, , transformations)\nsmolyak_basis(univariate_family, grid_kind, , transformations, M)\n\n\nCreate a sparse Smolyak basis using univariate_family (eg Chebyshev), which takes grid_kind and a dimension parameter. B > 0 caps the sum of blocks, while M > 0 caps blocks along each dimension separately, and transformations is a tuple of transformations applied coordinate-wise.\n\nExample\n\njulia> basis = smolyak_basis(Chebyshev, InteriorGrid(), Val(3),\n                             (BoundedLinear(2, 3), SemiInfRational(3.0, 4.0)))\nSparse multivariate basis on ℝ^2\n  Smolyak indexing, 3 total blocks, capped at 3, dimension 29\n  using Chebyshev polynomials (1st kind), interior grid, dimension: 9\n  transformations\n    (2.0,3.0) [linear transformation]\n    (3.0,∞) [rational transformation with scale 4.0]\n\njulia> dimension(basis)\n29\n\njulia> domain(basis)\n((2.0, 3.0), (3.0, Inf))\n\n\n\n\n\n","category":"function"},{"location":"#Using-bases","page":"SpectralKit","title":"Using bases","text":"","category":"section"},{"location":"#Introspection","page":"SpectralKit","title":"Introspection","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"is_function_basis\ndimension\ndomain","category":"page"},{"location":"#SpectralKit.is_function_basis","page":"SpectralKit","title":"SpectralKit.is_function_basis","text":"is_function_basis(F)\n\nis_function_basis(f::F)\n\nTest if the argument is a function basis, supporting the following interface:\n\ndomain for querying the domain,\ndimension for the dimension,\nbasis_at for function evaluation,\ngrid to obtain collocation points.\n\nlinear_combination and collocation_matrix are also supported, building on the above.\n\nCan be used on both types (preferred) and values (for convenience).\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.dimension","page":"SpectralKit","title":"SpectralKit.dimension","text":"dimension(basis)\n\nReturn the dimension of basis, a positive Int.\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.domain","page":"SpectralKit","title":"SpectralKit.domain","text":"domain(basis)\n\nThe domain of a function basis. A tuple of numbers (of arbitrary type, but usually Float64), or a tuple the latter.\n\n\n\n\n\n","category":"function"},{"location":"#Evaluation","page":"SpectralKit","title":"Evaluation","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"basis_at\nlinear_combination","category":"page"},{"location":"#SpectralKit.basis_at","page":"SpectralKit","title":"SpectralKit.basis_at","text":"basis_at(basis, x)\n\nReturn an iterable with known element type and length (Base.HasEltype(), Base.HasLength()) of basis functions in basis evaluated at x.\n\nMethods are type stable.\n\nnote: Note\nConsequences are undefined when evaluating outside the domain.\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.linear_combination","page":"SpectralKit","title":"SpectralKit.linear_combination","text":"linear_combination(basis, θ, x)\n\n\nEvaluate the linear combination of  θₖfₖ(x) of function basis f₁  at x, for the given order.\n\nThe length of θ should equal dimension(θ).\n\n\n\n\n\nlinear_combination(basis, θ)\n\n\nReturn a callable that calculates linear_combination(basis, θ, x) when called with x.\n\n\n\n\n\n","category":"function"},{"location":"#Grids-and-collocation","page":"SpectralKit","title":"Grids and collocation","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"grid\ncollocation_matrix","category":"page"},{"location":"#SpectralKit.grid","page":"SpectralKit","title":"SpectralKit.grid","text":"grid([T], basis)\n\nReturn a grid recommended for collocation, with dimension(basis) elements.\n\nT is used as a hint for the element type of grid coordinates, and defaults to Float64. The actual type can be broadened as required. Methods are type stable.\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.collocation_matrix","page":"SpectralKit","title":"SpectralKit.collocation_matrix","text":"collocation_matrix(basis, x)\n\n\nConvenience function to obtain a collocation matrix at gridpoints x, which is assumed to have a concrete eltype.\n\nMethods are type stable.\n\n\n\n\n\n","category":"function"},{"location":"#Internals","page":"SpectralKit","title":"Internals","text":"","category":"section"},{"location":"#Simplified-API-for-adding-custom-transformations","page":"SpectralKit","title":"Simplified API for adding custom transformations","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"SpectralKit.UnivariateTransformation\nSpectralKit.to_domain\nSpectralKit.from_domain","category":"page"},{"location":"#SpectralKit.UnivariateTransformation","page":"SpectralKit","title":"SpectralKit.UnivariateTransformation","text":"abstract type UnivariateTransformation\n\nAn abstract type for univariate transformations. Transformations are not required to be subtypes, this just documents the interface they need to support:\n\nto_domain\nfrom_domain\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.to_domain","page":"SpectralKit","title":"SpectralKit.to_domain","text":"to_domain(transformation, parent, x)\n\n!!! FIXME     document, especially differentiability requirements at infinite endpoints\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.from_domain","page":"SpectralKit","title":"SpectralKit.from_domain","text":"from_domain(transformation, parent, x)\n\n!!! FIXME     document, especially differentiability requirements at infinite endpoints\n\n\n\n\n\n","category":"function"},{"location":"#Grid-internals","page":"SpectralKit","title":"Grid internals","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"SpectralKit.gridpoint","category":"page"},{"location":"#SpectralKit.gridpoint","page":"SpectralKit","title":"SpectralKit.gridpoint","text":"gridpoint(basis, i)\n\n\nReturn a gridpoint for collocation, with 1 ≤ i ≤ dimension(basis).\n\nT is used as a hint for the element type of grid coordinates, and defaults to Float64. The actual type can be broadened as required. Methods are type stable.\n\nnote: Note\nNot all grids have this method defined, especially if it is impractical. See grid, which is part of the API, this function isn't.\n\n\n\n\n\n","category":"function"}]
}
