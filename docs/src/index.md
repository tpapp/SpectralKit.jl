# SpectralKit

This is a very simple package for *building blocks* of spectral methods. Its intended audience is users who are familiar with the theory and practice of these methods, and prefer to assemble their code from modular building blocks, potentially reusing calculations. If you need an introduction, a book like *Boyd (2001): Chebyshev and Fourier spectral methods* is a good place to start.

This package was designed primarily for solving functional equations, as usually encountered in economics when solving discrete and continuous-time problems. Key features include

1. evaluation of univariate and multivatiate basis functions, including Smolyak combinations,
2. transformed to the relevant domains of interest, eg ``[a,b] × [0,∞)``,
3. (partial) derivatives, with correct limits at endpoints,
4. allocation-free, thread safe linear combinations for the above with a given set of coefficients,
5. using [static arrays](https://github.com/JuliaArrays/StaticArrays.jl) extensively to avoid allocation and unroll *some* loops.

While there is some functionality in this package to *fit* approximations to existing functions, it does not use optimized algorithms (DCT) for that, as it was optimized for mapping a set of coefficients to residuals of functional equations at gridpoints.

Also, while the package should interoperate seamlessly with most AD frameworks, only the derivative API (explained below) is guaranteed to have correct derivatives of limits near infinity.

## Concepts

In this package,

1. A *basis* is a finite family of functions for approximating other functions. The [`dimension`](@ref) of a basis tells you how many functions are in there, while [`domain`](@ref) can be used to query its domain.

2. A [`grid`](@ref) is vector of *suggested* gridpoints for evaluating the function to be approximated that has useful theoretical properties. You can contruct a [`collocation_matrix`](@ref) using this grid (or any other set of points). Grids are associated with bases at the time of their construction: a basis with the same set of functions can have different grids.

3. [`basis_at`](@ref) returns an *iterator* for evaluating basis functions at an arbitrary point inside their domain. This iterator is meant to be heavily optimized and non-allocating. [`linear_combination`](@ref) is a convenience wrapper for obtaining a linear combination of basis functions at a given point.

## Univariate basis example

We construct a basis of Chebyshev polynomials on ``[0, 4]``. This requires a transformation since their canonical domain is ``[-1,1]``. Other transformations include [`SemiInfRational`](@ref) for ``[A, \infty]`` intervals  and [`InfRational`](@ref) for ``[-\infty, \infty] `` intervals.

We display the domian and the dimension (number of basis functions).
```@repl univariate
using SpectralKit
basis = Chebyshev(InteriorGrid(), 5) ∘ BoundedLinear(0, 4)
domain(basis)
dimension(basis)
```

We have chosen an interior grid, shown below. We `collect` the result for the purpose of this tutorial, since `grid` returns an iterable to avoid allocations.
```@repl univariate
collect(grid(basis))
```

We can show evaluate the basis functions at a given point. Again, it is an iterable, so we `collect` to show it here.
```@repl univariate
collect(basis_at(basis, 0.41))
```

We can evaluate linear combination as directly, or via partial application.
```@repl univariate
θ = [1, 0.5, 0.2, 0.3, 0.001];           # a vector of coefficients
x = 0.41
linear_combination(basis, θ, x)          # combination at some value
linear_combination(basis, θ)(x)          # also as a callable
```

We can also evaluate *derivatives* of either the basis or the linear combination at a given point. Here we want the derivatives up to order 3.
```@repl univariate
dx = (𝑑^2)(x)
collect(basis_at(basis, dx))
fdx = linear_combination(basis, θ, dx)
fdx[0] # the value
fdx[1] # the first derivative
```

Having an approximation, we can embed it in a larger basis, extending the coefficients accordingly.
```@repl univariate
basis2 = Chebyshev(EndpointGrid(), 8) ∘ transformation(basis)       # 8 Chebyshev polynomials
is_subset_basis(basis, basis2)              # we could augment θ …
augment_coefficients(basis, basis2, θ)      # … so let's do it
```

## Multivariate (Smolyak) approximation example

We set up a Smolyak basis to approximate functions on ``[-1,2] \times [-3, \infty]``, where the second dimension has a scaling of ``3``.

```@example smolyak
using SpectralKit, StaticArrays
basis = smolyak_basis(Chebyshev, InteriorGrid2(), SmolyakParameters(3), 2)
ct = coordinate_transformations(BoundedLinear(-1, 2.0), SemiInfRational(-3.0, 3.0))
basis_t = basis ∘ ct
```
Note how the basis can be combined with a transformation using `∘`.

We will approximate the following function:
```@example smolyak
f2((x1, x2)) = exp(x1) + exp(-abs2(x2))
```

We find the coefficients by solving with the collocation matrix.
```@example
θ = collocation_matrix(basis_t) \ f2.(grid(basis_t))
```

Finally, we check the approximation at a point.
```@example
z = (0.5, 0.7)                            # evaluate at this point
isapprox(f2(z), linear_combination(basis_t, θ)(z), rtol = 0.005)
```

## Infinite endpoints

Values and derivatives at ``\pm\infty`` should provide the correct limits.
```@repl
basis = Chebyshev(InteriorGrid(), 4) ∘ InfRational(0.0, 1.0)
collect(basis_at(basis, 𝑑(Inf)))
collect(basis_at(basis, 𝑑(-Inf)))
```

## Constructing bases

### Grid specifications

```@docs
EndpointGrid
InteriorGrid
InteriorGrid2
```

### Domains and transformations

A transformation maps values between a *domain*, usually specified by
the basis, and the (co)domain that is specified by a transformation.
Transformations are not required to be subtypes of anything, but need
to support

```@docs
transform_to
transform_from
domain
```

In most cases you do not need to specify a domain directly: transformations specify their domains (eg from ``(0, ∞)``), and the codomain is determined by a basis. However, the following can be used to construct and query some concrete domains.

```@docs
domain_kind
coordinate_domains
```

Bases are defined on a *canonical domain*, such as ``[-1, 1]`` for Chebyshev polynomials. *Transformations* map other uni- and multivariate sets into these domains.

```@docs
BoundedLinear
InfRational
SemiInfRational
coordinate_transformations
```

### Univariate bases

Currently, only Chebyshev polynomials are implemented. Univariate bases operate on real numbers.

```@docs
Chebyshev
```

### Multivariate bases

Multivariate bases operate on tuples or vectors (`StaticArrays.SVector` is preferred for performance, but all `<:AbstractVector` types should work).

```@docs
SmolyakParameters
smolyak_basis
```

## Using bases

### Introspection

```@docs
is_function_basis
dimension
transformation
```

See also [`domain`](@ref).

### Evaluation

```@docs
basis_at
linear_combination
```

### Grids and collocation

```@docs
grid
collocation_matrix
```

### Augment coordinates for a wider basis

```@docs
augment_coefficients
is_subset_basis
```

## Derivatives

!!! note
    API for derivatives is still experimental and subject to change.

For univariate functions, use [`𝑑`](@ref). For multivariate functions, use partial derivatives with [`∂`](@ref).

```@docs
𝑑
∂
```

## Internals

This section of the documentation is probably only relevant to contributors and others who want to understand the internals.

### Type hierarchies

Generally, the abstract types below are not part of the exposed API, and new types don't have to subtype them (unless they want to rely on the existing convenience methods). They are merely for code organization.

```@docs
SpectralKit.AbstractUnivariateDomain
```

### Grid internals

```@docs
SpectralKit.gridpoint
```
