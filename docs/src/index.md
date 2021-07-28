# SpectralKit

This is a very simple package for *building blocks* of spectral methods. Its intended audience is users who are familiar with the theory and practice of these methods, and prefer to assemble their code from modular building blocks. If you need an introduction, a book like *Boyd (2001): Chebyshev and Fourier spectral methods* is a good place to start.

The package is optimized for solving functional equations, as usually encountered in economics when solving discrete and continuous-time problems. It uses [static arrays](https://github.com/JuliaArrays/StaticArrays.jl) extensively to avoid allocation and unroll *some* loops. Key functionality includes evaluating a set of basis functions, their linear combination at arbitrary points in a fast manner, for use in threaded code. These should work seamlessly with automatic differentiation frameworks when derivatives are needed.

## Introduction

In this package,

1. A *basis* is a finite family of functions for approximating other functions. The [`dimension`](@ref) of a basis tells you how many functions are in there, while [`domain`](@ref) can be used to query its domain.

2. A [`grid`](@ref) is vector of *suggested* gridpoints for evaluating the function to be approximated that has useful theoretical properties. You can contruct a [`collocation_matrix`](@ref) using this grid. Grids are associated with bases at the time of their construction: a basis with the same set of functions can have different grids.

3. [`basis_at`](@ref) returns an *iterator* for evaluating basis functions at an arbitrary point inside their domain. This iterator is meant to be heavily optimized and non-allocating. [`linear_combination`](@ref) is a convenience wrapper for obtaining a linear combination of basis functions at a given point.

A basis is constructed using

1. a primitive family on a *fixed* domain, eg [`Chebyshev`](@ref),

2. a grid specification like [`InteriorGrid`](@ref),

3. a number of parameters that determine the number of basis functions.

4. transformation(s) that project the domain of a primitive family into the domain.

The following example provides an overview of the interface for univariate function families:
```@repl
using SpectralKit
basis = univariate_basis(Chebyshev, EndpointGrid(), # 5 Chebyshev polynomials on [0,2]
                         5, BoundedLinear(0, 2))
is_function_basis(basis) # tells us that we support the interface below
dimension(basis) # number of basis functions
domain(basis)            # domain
grid(basis) # Gauss-Lobatto grid
collect(basis_at(basis, 0.41)) # iterator for basis functions at 0.41
θ = [1, 0.5, 0.2, 0.3, 0.001] # a vector of coefficients
linear_combination(basis, θ, 0.41) # combination at some value
linear_combination(basis, θ)(0.41) # also as a callable
```

## Constructing bases

### Primitive families

Currently, only Chebyshev polynomials are implemented.

```@docs
Chebyshev
```

### Grid specifications

```@docs
InteriorGrid
EndpointGrid
```

### Transformations

```@docs
BoundedLinear
InfRational
SemiInfRational
```

### Univariate bases

```@docs
univariate_basis
```

### Multivariate bases

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

## Internals

This section of the documentation is probably only relevant to contributors and others who want to understand the internals.

### Simplified API for adding custom transformations

```@docs
SpectralKit.UnivariateTransformation
SpectralKit.to_domain
SpectralKit.from_domain
```

### Grid internals

```@docs
SpectralKit.gridpoint
```
