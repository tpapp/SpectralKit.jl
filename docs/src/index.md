# SpectralKit

This is a very simple package for *building blocks* of spectral methods. Its intended audience is users who are familiar with the theory and practice of these methods, and prefer to assemble their code from modular building blocks. If you need an introduction, a book like *Boyd (2001): Chebyshev and Fourier spectral methods* is a good place to start.

The package is optimized for solving functional equations, as usually encountered in economics when solving discrete and continuous-time problems. It uses [static arrays](https://github.com/JuliaArrays/StaticArrays.jl) extensively to avoid allocation and unroll *some* loops. Key functionality includes evaluating a set of basis functions, their linear combination at arbitrary points in a fast manner, for use in threaded code. These should work seamlessly with automatic differentiation frameworks, but also has its own primitives for obtaining derivatives of basis functions.

## Introduction

In this package,

1. A *basis* is a finite family of functions for approximating other functions. The [`dimension`](@ref) of a basis tells you how many functions are in there, while [`domain`](@ref) can be used to query its domain.

2. A [`grid`](@ref) is vector of *suggested* gridpoints for evaluating the function to be approximated that has useful theoretical properties. You can contruct a [`collocation_matrix`](@ref) using this grid (or any other set of points). Grids are associated with bases at the time of their construction: a basis with the same set of functions can have different grids.

3. [`basis_at`](@ref) returns an *iterator* for evaluating basis functions at an arbitrary point inside their domain. This iterator is meant to be heavily optimized and non-allocating. [`linear_combination`](@ref) is a convenience wrapper for obtaining a linear combination of basis functions at a given point.

A basis is constructed using

1. a family on a *fixed* domain, eg [`Chebyshev`](@ref),

2. a grid specification like [`InteriorGrid`](@ref),

3. a number of parameters that determine the number of basis functions.

A set of coordinates for a particular basis can be augmented for a wider basis with [`augment_coefficients`](@ref).

Currenly, all bases have the domain ``[-1,1]`` or ``[-1,1]^n``. Facilities are provided for coordinatewise *transformations* to other domains.

## Examples

### Univariate family on `[-1,1]`

```@repl
using SpectralKit
basis = Chebyshev(EndpointGrid(), 5)        # 5 Chebyshev polynomials
is_function_basis(basis)                    # ie we support the interface below
dimension(basis)                            # number of basis functions
domain(basis)                               # domain
grid(basis)                                 # Gauss-Lobatto grid
collect(basis_at(basis, 0.41))              # iterator for basis functions at 0.41
collect(basis_at(basis, derivatives(0.41))) # values and 1st derivatives
θ = [1, 0.5, 0.2, 0.3, 0.001]               # a vector of coefficients
linear_combination(basis, θ, 0.41)          # combination at some value
linear_combination(basis, θ)(0.41)          # also as a callable
basis2 = Chebyshev(EndpointGrid(), 8)       # 8 Chebyshev polynomials
is_subset_basis(basis, basis2)              # we could augment θ …
augment_coefficients(basis, basis2, θ)      # … so let's do it
```

### Smolyak approximation on a transformed domain

```@repl
using SpectralKit, StaticArrays
function f2(x)                  # bivariate function we approximate
    x1, x2 = x                  # takes vectors
    exp(x1) + exp(-abs2(x2))
end
ct = coordinate_transformations(BoundedLinear(-1, 2.0), SemiInfRational(-3.0, 3.0))
basis = smolyak_basis(Chebyshev, InteriorGrid2(), SmolyakParameters(3), 2)
x = grid(basis)
θ = collocation_matrix(basis) \ f2.(from_pm1.(ct, x)) # find the coefficients
z = (0.5, 0.7)                                        # evaluate at this point
isapprox(f2(z), linear_combination(basis, θ, to_pm1(ct, z)), rtol = 0.005)
```

## Constructing bases

### Grid specifications

```@docs
EndpointGrid
InteriorGrid
InteriorGrid2
```

### Univariate and multivariate transformations

Bases are defined on the domain ``[-1, 1]`` or ``[-1, 1]^n``. *Transformations* map other uni- and multivariate sets into these domains.

```@docs
to_pm1
from_pm1
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
domain
```

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

If derivatives along a coordinate are needed, use [`derivatives`](@ref). For multiple coordinates, the result will be nested in the order of increasing tags. When unspecified, tags are assigned automatically from left to right.

```@docs
derivatives
```

## Internals

This section of the documentation is probably only relevant to contributors and others who want to understand the internals.

### Simplified API for adding custom transformations

```@docs
SpectralKit.UnivariateTransformation
```

### Grid internals

```@docs
SpectralKit.gridpoint
```
