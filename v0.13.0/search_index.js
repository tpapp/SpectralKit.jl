var documenterSearchIndex = {"docs":
[{"location":"#SpectralKit","page":"SpectralKit","title":"SpectralKit","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"This is a very simple package for building blocks of spectral methods. Its intended audience is users who are familiar with the theory and practice of these methods, and prefer to assemble their code from modular building blocks, potentially reusing calculations. If you need an introduction, a book like Boyd (2001): Chebyshev and Fourier spectral methods is a good place to start.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"This package was designed primarily for solving functional equations, as usually encountered in economics when solving discrete and continuous-time problems. Key features include","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"evaluation of univariate and multivatiate basis functions, including Smolyak combinations,\ntransformed to the relevant domains of interest, eg ab  0),\n(partial) derivatives, with correct limits at endpoints,\nallocation-free, thread safe linear combinations for the above with a given set of coefficients,\nusing static arrays extensively to avoid allocation and unroll some loops.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"While there is some functionality in this package to fit approximations to existing functions, it is not ideal for that, as it was optimized for mapping a set of coefficients to residuals of functional equations at gridpoints.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"Also, while the package should interoperate seamlessly with most AD frameworks, only the derivative API (explained below) is guaranteed to have correct derivatives of limits near infinity.","category":"page"},{"location":"#Concepts","page":"SpectralKit","title":"Concepts","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"In this package,","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"A basis is a finite family of functions for approximating other functions. The dimension of a basis tells you how many functions are in there, while domain can be used to query its domain.\nA grid is vector of suggested gridpoints for evaluating the function to be approximated that has useful theoretical properties. You can contruct a collocation_matrix using this grid (or any other set of points). Grids are associated with bases at the time of their construction: a basis with the same set of functions can have different grids.\nbasis_at returns an iterator for evaluating basis functions at an arbitrary point inside their domain. This iterator is meant to be heavily optimized and non-allocating. linear_combination is a convenience wrapper for obtaining a linear combination of basis functions at a given point.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"A basis is constructed using","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"a family on a fixed domain, eg Chebyshev,\na grid specification like InteriorGrid,\na number of parameters that determine the number of basis functions.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"A set of coordinates for a particular basis can be augmented for a wider basis with augment_coefficients.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"Bases have a “canonical” domain, eg -11 or -11^n for Chebyshev polynomials. Use transformations for mapping to other domains.","category":"page"},{"location":"#Examples","page":"SpectralKit","title":"Examples","text":"","category":"section"},{"location":"#Univariate-family-on-[-1,1]","page":"SpectralKit","title":"Univariate family on [-1,1]","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"using SpectralKit\nbasis = Chebyshev(EndpointGrid(), 5)        # 5 Chebyshev polynomials\nis_function_basis(basis)                    # ie we support the interface below\ndimension(basis)                            # number of basis functions\ndomain(basis)                               # domain\ngrid(basis)                                 # Gauss-Lobatto grid\ncollect(basis_at(basis, 0.41))              # iterator for basis functions at 0.41\ncollect(basis_at(basis, derivatives(0.41))) # values and 1st derivatives\nθ = [1, 0.5, 0.2, 0.3, 0.001]               # a vector of coefficients\nlinear_combination(basis, θ, 0.41)          # combination at some value\nlinear_combination(basis, θ)(0.41)          # also as a callable\nbasis2 = Chebyshev(EndpointGrid(), 8)       # 8 Chebyshev polynomials\nis_subset_basis(basis, basis2)              # we could augment θ …\naugment_coefficients(basis, basis2, θ)      # … so let's do it","category":"page"},{"location":"#Smolyak-approximation-on-a-transformed-domain","page":"SpectralKit","title":"Smolyak approximation on a transformed domain","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"using SpectralKit, StaticArrays\nfunction f2(x)                  # bivariate function we approximate\n    x1, x2 = x                  # takes vectors\n    exp(x1) + exp(-abs2(x2))\nend\nct = coordinate_transformations(BoundedLinear(-1, 2.0), SemiInfRational(-3.0, 3.0))\nbasis = smolyak_basis(Chebyshev, InteriorGrid2(), SmolyakParameters(3), 2)\nx = grid(basis)\nθ = collocation_matrix(basis) \\ f2.(from_pm1.(ct, x)) # find the coefficients\nz = (0.5, 0.7)                                        # evaluate at this point\nisapprox(f2(z), (linear_combination(basis, θ) ∘ ct)(z), rtol = 0.005)","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"Note how the transformation can be combined with ∘ to a callable that evaluates a transformed linear combination at z.","category":"page"},{"location":"#Constructing-bases","page":"SpectralKit","title":"Constructing bases","text":"","category":"section"},{"location":"#Grid-specifications","page":"SpectralKit","title":"Grid specifications","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"EndpointGrid\nInteriorGrid\nInteriorGrid2","category":"page"},{"location":"#SpectralKit.EndpointGrid","page":"SpectralKit","title":"SpectralKit.EndpointGrid","text":"struct EndpointGrid <: SpectralKit.AbstractGrid\n\nGrid that includes endpoints (eg Gauss-Lobatto).\n\nnote: Note\nFor small dimensions may fall back to a grid that does not contain endpoints.\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.InteriorGrid","page":"SpectralKit","title":"SpectralKit.InteriorGrid","text":"struct InteriorGrid <: SpectralKit.AbstractGrid\n\nGrid with interior points (eg Gauss-Chebyshev).\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.InteriorGrid2","page":"SpectralKit","title":"SpectralKit.InteriorGrid2","text":"struct InteriorGrid2 <: SpectralKit.AbstractGrid\n\nGrid with interior points that results in smaller grids than InteriorGrid when nested. Equivalent to an EndpointGrid with endpoints dropped.\n\n\n\n\n\n","category":"type"},{"location":"#Domains-and-transformations","page":"SpectralKit","title":"Domains and transformations","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"A transformation maps values between a domain, usually specified by the basis, and the (co)domain that is specified by a transformation. Transformations are not required to be subtypes of anything, but need to support","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"transform_to\ntransform_from\ndomain","category":"page"},{"location":"#SpectralKit.transform_to","page":"SpectralKit","title":"SpectralKit.transform_to","text":"transform_to(domain, transformation, x)\n\nTransform x to domain using transformation.\n\n!!! FIXME     document, especially differentiability requirements at infinite endpoints\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.transform_from","page":"SpectralKit","title":"SpectralKit.transform_from","text":"transform_from(domain, transformation, x)\n\nTransform x from domain using transformation.\n\n!!! FIXME     document, especially differentiability requirements at infinite endpoints\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.domain","page":"SpectralKit","title":"SpectralKit.domain","text":"domain(basis)\n\nThe domain of a function basis.\n\ndomain(transformation)\n\nThe (co)domain of a transformation. The “other” domain (codomain, depending on the mapping) is provided explicitly for transformations, and should be compatible with thedomain of the basis.\n\nSee domain_kind for the interface supported by domains.\n\n\n\n\n\n","category":"function"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"In most cases you do not need to specify a domain directly: transformations specify their domains (eg from (0 )), and the codomain is determined by a basis. However, the following can be used to construct and query some concrete domains.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"domain_kind\ncoordinate_domains","category":"page"},{"location":"#SpectralKit.domain_kind","page":"SpectralKit","title":"SpectralKit.domain_kind","text":"domain_kind(x)\n\n\nReturn the kind of a domain type (preferred) or value. Errors for objects/types which are not domains. Also works for domains of transformations.\n\nThe following return values are possible:\n\n:univariate, the bounds of which can be accessed using minimum, maximum, and\n\nextrema,\n\n:multivariate, which supports length, getindex ([]), and conversion with Tuple.\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.coordinate_domains","page":"SpectralKit","title":"SpectralKit.coordinate_domains","text":"coordinate_domains(domains)\n\n\nCreate domains which are the product of univariate domains. The result support length, indexing with integers, and Tuple for conversion.\n\n\n\n\n\ncoordinate_domains(domains)\n\n\n\n\n\n\ncoordinate_domains(_, domain)\n\n\nCreate a coordinate domain which is the product of domain repeated N times.\n\n\n\n\n\ncoordinate_domains(N, domain)\n\n\n\n\n\n\n","category":"function"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"Bases are defined on a canonical domain, such as -1 1 for Chebyshev polynomials. Transformations map other uni- and multivariate sets into these domains.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"BoundedLinear\nInfRational\nSemiInfRational\ncoordinate_transformations","category":"page"},{"location":"#SpectralKit.BoundedLinear","page":"SpectralKit","title":"SpectralKit.BoundedLinear","text":"struct BoundedLinear{T<:Real} <: SpectralKit.AbstractUnivariateTransformation\n\nTransform the domain to y ∈ (a, b), using y = x  s + m.\n\nm and s are calculated and checked by the constructor; a < b is enforced.\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.InfRational","page":"SpectralKit","title":"SpectralKit.InfRational","text":"InfRational(A, L)\n\n\nThe domain transformed to (-Inf, Inf) using y = A + L  x  (1 - x^2), with L > 0.\n\nExample mappings (for domain (-11))\n\n0  A\n05  A  L  3\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.SemiInfRational","page":"SpectralKit","title":"SpectralKit.SemiInfRational","text":"SemiInfRational(A, L)\n\n\nThe domian transformed to  [A, Inf) (when L > 0) or (-Inf,A] (when L < 0) using y = A + L  (1 + x)  (1 - x).\n\nWhen used with Chebyshev polynomials, also known as a “rational Chebyshev” basis.\n\nExample mappings for the domain (-11)\n\n-12  A + L  3\n0  A + L\n12  A + 3  L\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.coordinate_transformations","page":"SpectralKit","title":"SpectralKit.coordinate_transformations","text":"coordinate_transformations(transformations)\n\n\nWrapper for coordinate-wise transformations.\n\njulia> using StaticArrays\n\njulia> ct = coordinate_transformations(BoundedLinear(0, 2), SemiInfRational(2, 3))\ncoordinate transformations\n  (0.0,2.0) ↔ domain [linear transformation]\n  (2,∞) ↔ domain [rational transformation with scale 3]\n\njulia> d1 = domain(Chebyshev(InteriorGrid(), 5))\n[-1,1]\n\njulia> dom = coordinate_domains(d1, d1)\n[-1,1]²\n\njulia> x = transform_from(dom, ct, (0.4, 0.5))\n(1.4, 11.0)\n\njulia> y = transform_to(dom, ct, x)\n(0.3999999999999999, 0.5)\n\n\n\n\n\n","category":"function"},{"location":"#Univariate-bases","page":"SpectralKit","title":"Univariate bases","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"Currently, only Chebyshev polynomials are implemented. Univariate bases operate on real numbers.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"Chebyshev","category":"page"},{"location":"#SpectralKit.Chebyshev","page":"SpectralKit","title":"SpectralKit.Chebyshev","text":"struct Chebyshev{K<:SpectralKit.AbstractGrid} <: SpectralKit.UnivariateBasis\n\nThe first N Chebyhev polynomials of the first kind, defined on [-1,1].\n\n\n\n\n\n","category":"type"},{"location":"#Multivariate-bases","page":"SpectralKit","title":"Multivariate bases","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"Multivariate bases operate on tuples or vectors (StaticArrays.SVector is preferred for performance, but all <:AbstractVector types should work).","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"SmolyakParameters\nsmolyak_basis","category":"page"},{"location":"#SpectralKit.SmolyakParameters","page":"SpectralKit","title":"SpectralKit.SmolyakParameters","text":"SmolyakParameters(B)\nSmolyakParameters(B, M)\n\n\nParameters for Smolyak grids that are independent of the dimension of the domain.\n\nPolynomials are organized into blocks (of eg 1, 2, 2, 4, 8, 16, …) polynomials (and corresponding gridpoints), indexed with a block index b that starts at 0. B ≥ ∑ bᵢ and 0 ≤ bᵢ ≤ M constrain the number of blocks along each dimension i.\n\nM > B is not an error, but will be normalized to M = B with a warning.\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.smolyak_basis","page":"SpectralKit","title":"SpectralKit.smolyak_basis","text":"smolyak_basis(\n    univariate_family,\n    grid_kind,\n    smolyak_parameters,\n    _\n)\n\n\nCreate a sparse Smolyak basis.\n\nArguments\n\nunivariate_family: should be a callable that takes a grid_kind and a dimension parameter, eg Chebyshev.\ngrid_kind: the grid kind, eg InteriorGrid() etc.\nsmolyak_parameters: the Smolyak grid specification parameters, see SmolyakParameters.\nN: the dimension. wrapped in a Val for type stability, a convenience constructor also takes integers.\n\nExample\n\njulia> basis = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(3), 2)\nSparse multivariate basis on ℝ²\n  Smolyak indexing, ∑bᵢ ≤ 3, all bᵢ ≤ 3, dimension 81\n  using Chebyshev polynomials (1st kind), InteriorGrid(), dimension: 27\n\njulia> dimension(basis)\n81\n\njulia> domain(basis)\n[-1,1]²\n\nProperties\n\nGrids nest: increasing arguments of SmolyakParameters result in a refined grid that contains points of the cruder grid.\n\n\n\n\n\n","category":"function"},{"location":"#Using-bases","page":"SpectralKit","title":"Using bases","text":"","category":"section"},{"location":"#Introspection","page":"SpectralKit","title":"Introspection","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"is_function_basis\ndimension","category":"page"},{"location":"#SpectralKit.is_function_basis","page":"SpectralKit","title":"SpectralKit.is_function_basis","text":"is_function_basis(::Type{F})\n\nis_function_basis(f::F)\n\nTest if the argument (value or type) is a function basis, supporting the following interface:\n\ndomain for querying the domain,\ndimension for the dimension,\nbasis_at for function evaluation,\ngrid to obtain collocation points.\n\nlinear_combination and collocation_matrix are also supported, building on the above.\n\nCan be used on both types (preferred) and values (for convenience).\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.dimension","page":"SpectralKit","title":"SpectralKit.dimension","text":"dimension(basis)\n\nReturn the dimension of basis, a positive Int.\n\n\n\n\n\n","category":"function"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"See also domain.","category":"page"},{"location":"#Evaluation","page":"SpectralKit","title":"Evaluation","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"basis_at\nlinear_combination","category":"page"},{"location":"#SpectralKit.basis_at","page":"SpectralKit","title":"SpectralKit.basis_at","text":"basis_at(basis, x)\n\nReturn an iterable with known element type and length (Base.HasEltype(), Base.HasLength()) of basis functions in basis evaluated at x.\n\nUnivariate bases operate on real numbers, while for multivariate bases, Tuples or StaticArrays.SVector are preferred for performance, though all <:AbstractVector types should work.\n\nMethods are type stable.\n\nnote: Note\nConsequences are undefined when evaluating outside the domain.\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.linear_combination","page":"SpectralKit","title":"SpectralKit.linear_combination","text":"linear_combination(basis, θ, x)\n\n\nEvaluate the linear combination of  θₖfₖ(x) of function basis f₁  at x, for the given order.\n\nThe length of θ should equal dimension(θ).\n\n\n\n\n\nlinear_combination(basis, θ)\n\n\nReturn a callable that calculates linear_combination(basis, θ, x) when called with x.\n\nUse linear_combination(basis, θ) ∘ transformation for domain transformations.\n\n\n\n\n\n","category":"function"},{"location":"#Grids-and-collocation","page":"SpectralKit","title":"Grids and collocation","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"grid\ncollocation_matrix","category":"page"},{"location":"#SpectralKit.grid","page":"SpectralKit","title":"SpectralKit.grid","text":"grid([T], basis)\n\nReturn an iterator for the grid recommended for collocation, with dimension(basis) elements.\n\nT for the element type of grid coordinates, and defaults to Float64. Methods are type stable.\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.collocation_matrix","page":"SpectralKit","title":"SpectralKit.collocation_matrix","text":"collocation_matrix(basis)\ncollocation_matrix(basis, x)\n\n\nConvenience function to obtain a “collocation matrix” at points x, which is assumed to have a concrete eltype. The default is x = grid(basis), specialized methods may exist for this when it makes sense.\n\nThe collocation matrix may not be an AbstractMatrix, all it needs to support is C \\ y for compatible vectors y = f.(x).\n\nMethods are type stable. The elements of x can be derivatives.\n\n\n\n\n\n","category":"function"},{"location":"#Augment-coordinates-for-a-wider-basis","page":"SpectralKit","title":"Augment coordinates for a wider basis","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"augment_coefficients\nis_subset_basis","category":"page"},{"location":"#SpectralKit.augment_coefficients","page":"SpectralKit","title":"SpectralKit.augment_coefficients","text":"augment_coefficients(basis1, basis2, θ1)\n\nReturn a set of coefficients θ2 for basis2 such that\n\nlinear_combination(basis1, θ1, x) == linear_combination(basis2, θ2, x)\n\nfor any x in the domain. In practice this means padding with zeros.\n\nThrow a ArgumentError if the bases are incompatible with each other or x, or this is not possible. Methods may not be defined for incompatible bases, compatibility between bases can be checked with is_subset_basis.\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.is_subset_basis","page":"SpectralKit","title":"SpectralKit.is_subset_basis","text":"is_subset_basis(basis1, basis2)\n\n\nReturn a Bool indicating whether coefficients in basis1 can be augmented to basis2 with augment_coefficients.\n\nnote: Note\ntrue does not mean that coefficients from basis1 can just be padded with zeros, since they may be in different positions. Always use augment_coefficients.\n\n\n\n\n\n","category":"function"},{"location":"#Derivatives","page":"SpectralKit","title":"Derivatives","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"note: Note\nAPI for derivatives is still experimental and subject to change.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"For univariate functions, use derivatives. For multivariate functions, use partial derivatives with ∂.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"derivatives\n∂","category":"page"},{"location":"#SpectralKit.derivatives","page":"SpectralKit","title":"SpectralKit.derivatives","text":"derivatives(x, ::Val(N) = Val(1))\n\nObtain N derivatives (and the function value) at a scalar x. The ith derivative can be accessed with [i] from results, with [0] for the function value.\n\nImportant note about transformations\n\nAlways use derivatives before a transformation for correct results. For example, for some transformation t and value x in the transformed domain,\n\n# right\nlinear_combination(basis, θ, transform_to(domain(basis), t, derivatives(x)))\n# right (convenience form)\n(linear_combination(basis, θ) ∘ t)(derivatives(x))\n\ninstead of\n\n# WRONG\nlinear_combination(basis, θ, derivatives(transform_to(domain(basis), t, x)))\n\nFor multivariate calculations, use the ∂ interface.\n\nExample\n\njulia> basis = Chebyshev(InteriorGrid(), 3)\nChebyshev polynomials (1st kind), InteriorGrid(), dimension: 3\n\njulia> C = collect(basis_at(basis, derivatives(0.1)))\n3-element Vector{SpectralKit.Derivatives{2, Float64}}:\n 1.0 + 0.0⋅Δ\n 0.1 + 1.0⋅Δ\n -0.98 + 0.4⋅Δ\n\njulia> C[1][1]                         # 1st derivative of the linear term is 1\n0.0\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.∂","page":"SpectralKit","title":"SpectralKit.∂","text":"∂(_, partials)\n\n\nPartial derivative specification. The first argument is Val(::Int) or simply an Int (for convenience, using constant folding), determining the dimension of the argument.\n\nSubsequent arguments are indices of the input variable.\n\njulia> ∂(3, (), (1, 1), (2, 3))\npartial derivatives\n[1] f\n[2] ∂²f/∂²x₁\n[3] ∂²f/∂x₂∂x₃\n\n\n\n\n\n∂(∂specification, x)\n\n\nInput wrappert type for evaluating partial derivatives ∂specification at x.\n\njulia> using StaticArrays\n\njulia> s = ∂(Val(2), (), (1,), (2,), (1, 2))\npartial derivatives\n[1] f\n[2] ∂f/∂x₁\n[3] ∂f/∂x₂\n[4] ∂²f/∂x₁∂x₂\n\njulia> ∂(s, SVector(1, 2))\npartial derivatives\n[1] f\n[2] ∂f/∂x₁\n[3] ∂f/∂x₂\n[4] ∂²f/∂x₁∂x₂\nat [1, 2]\n\n\n\n\n\n∂(x, partials)\n\n\nShorthand for ∂(x, ∂(Val(length(x)), partials...)). Ideally needs an SVector or a Tuple so that size information can be obtained statically.\n\n\n\n\n\n","category":"function"},{"location":"#Internals","page":"SpectralKit","title":"Internals","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"This section of the documentation is probably only relevant to contributors and others who want to understand the internals.","category":"page"},{"location":"#Type-hierarchies","page":"SpectralKit","title":"Type hierarchies","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"Generally, the abstract types below are not part of the exposed API, and new types don't have to subtype them (unless they want to rely on the existing convenience methods). They are merely for code organization.","category":"page"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"SpectralKit.AbstractUnivariateDomain","category":"page"},{"location":"#SpectralKit.AbstractUnivariateDomain","page":"SpectralKit","title":"SpectralKit.AbstractUnivariateDomain","text":"Univariate domain representation. Supports extrema, minimum, maximum.\n\nnote: Note\nImplementations only need to define extrema.\n\n\n\n\n\n","category":"type"},{"location":"#Grid-internals","page":"SpectralKit","title":"Grid internals","text":"","category":"section"},{"location":"","page":"SpectralKit","title":"SpectralKit","text":"SpectralKit.gridpoint","category":"page"},{"location":"#SpectralKit.gridpoint","page":"SpectralKit","title":"SpectralKit.gridpoint","text":"gridpoint(_, basis, i)\n\n\nReturn a gridpoint for collocation, with 1 ≤ i ≤ dimension(basis).\n\nT is used as a hint for the element type of grid coordinates. The actual type can be broadened as required. Methods are type stable.\n\nnote: Note\nNot all grids have this method defined, especially if it is impractical. See grid, which is part of the API, this function isn't.\n\n\n\n\n\n","category":"function"}]
}
