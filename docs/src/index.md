# SpectralKit

This is a very simple package for *building blocks* of spectral methods. Its intended audience is users who are familiar with the theory and practice of these methods, and prefer to assemble their code from modular building blocks.

The following provides an overview of the interface for univariate function families on various domains:
```@repl
using SpectralKit
basis = univariate_basis(Chebyshev(5), # Chebyshev polynomials on [0,2]
                         BoundedLinear(0, 2))
is_function_basis(basis) # tells us that we support the interface below
dimension(basis) # number of basis functions
domain(basis) # endpoints 
grid(basis, InteriorGrid()) # Gauss-Chebyshev grid
grid(basis, EndpointGrid()) # Gauss-Lobatto grid
collect(basis_at(F, 0.41)) # iterator for basis functions at 0.41
θ = [1 0.5 0.2 0.3 0.001] # a vector of coefficients
linear_combination(basis, θ, 0.41) # combination at some value
linear_combination(basis, θ)(0.41) # also as a callable
```

## Abstract interface for function families

### Bases generics

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
InteriorGrid
EndpointGrid
collocation_matrix
```

## Specific function families

### Building blocks

```@docs
Chebyshev
```

### Transformations

```@docs
univariate_basis
BoundedLinear
InfRational
SemiInfRational
```

### Simplified API for adding custom transformations

```@docs
SpectralKit.UnivariateTransformation
SpectralKit.to_domain
SpectralKit.from_domain
```
