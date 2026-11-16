# Overview

```@docs
SpectralKit
```

This page introduces the concepts of the package through univariate examples.

## Concepts

For the purposes of this package,

1. A *basis* is a finite *family* of functions, defined over a given domain.

2. Each family has a canonical domain: for example, Chebyshev polynomials are defined over ``[-1,1]``. This is mostly hidden from the user, allowing the specification of custom domains using *transformations*.

3. The number of basis functions is the *dimension*.

4. Each basis has an associated *grid* recommended for collocation methods.

5. The dimension is determined by the *level* and the grid *kind*.

6. The grid also has a level, which defaults to the level of the basis: default grid has as many points as the dimension of the basis.

7. Levels allow *nesting* gridpoints. Increasing the level (or the grid level) of the basis adds more grid points to existing ones.

## Univariate bases

Let's demonstrate these concepts with a concrete example, by constructing a basis of Chebyshev polynomials, transformed to the interval ``[0, 1]`` (using a linear mapping), with level = 2.

```@repl univariate
using SpectralKit;
basis = UnivariateBasis(Chebyshev(), Interior(), BoundedLinear(0, 1), 2)
```

The default printing methods shows the dimension, but one can of query it too using the API:

```@repl univariate
dimension(basis)
```

The grid is an iterable calculated *on demand*, so that `grid` does not allocate:
```@repl univariate
grid(basis)
```

To get the number of points in a grid, just use `grid(length(basis))`.

For univariate bases, levels are integers starting from `0`. One could use a denser grid, eg for the a least squares approximation, by also specifying *a level for the grid*:
```@repl univariate
basis_ls = UnivariateBasis(Chebyshev(), Interior(), BoundedLinear(0, 1), 1, 2)
```

Increasing the level adds more points. Grids can be calculated with types other than `Float64` (the default), here we use `Float16` for more compact printing:
```@repl univariate
[grid(Float16, UnivariateBasis(Chebyshev(), Interior(), BoundedLinear(0, 1), level))
 for level in 0:3]
```

## Grid kinds

Currently, two kinds of grids are supported: [`Endpoints`](@ref) and [`Interior`](@ref).

`Endpoints()` grids are Chebyshev nodes of the second kind, also known as Chebyshev–Lobatto nodes or Chebyshev extrema. They are the [extrema of Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_nodes), and nest with each level adding 2, 2, 4, 8, … gridpoints:

```@repl gridkinds
using SpectralKit
[grid(Float16, UnivariateBasis(Chebyshev(), Endpoints(), BoundedLinear(-1, 1), level))
 for level in 0:3]
```

`Interior()` grids are these with -1 and 1 dropped:

```@repl gridkinds
[grid(Float16, UnivariateBasis(Chebyshev(), Endpoints(), BoundedLinear(-1, 1), level))
 for level in 0:3]
```

## Domain transformations

The domain transformations are [`BoundedLinear`](@ref), [`SemiInfRational`](@ref), and [`InfRational`](@ref), with the latter transforming to semi-infinite regions, such as ``(0, ∞)``, and the real line, respectively. The latter two have parameters that influence the scale:

```@repl gridkinds
[grid(Float16, UnivariateBasis(Chebyshev(), Interior(), InfRational(; scale), 2))
 for scale in [0.5, 1.0, 2.0]]
```

and location

```@repl gridkinds
[grid(Float16, UnivariateBasis(Chebyshev(), Interior(),
                               SemiInfRational(; endpoint), 2))
 for endpoint in 0:2]
```

## Linear combinations

There are two ways to *evaluate* a combination of basis functions: either use [`linear_combination`](@ref) with a basis and a vector of coefficients, which gives a callable inside the domain, or to add the input as the third argument.

```@repl linear_combination
using SpectralKit, Miter
basis = UnivariateBasis(Chebyshev(), Interior(), InfRational(), 2)
θ = [1/x^2 for x in 1:7]
linear_combination(basis, θ, 0.5)
```

```@example linear_combination
let f = linear_combination(basis, θ), g = grid(basis)
    Plot(Lines((x, f(x)) for x in range(extrema(g)...; length = 500)),
         Scatter(MarkSymbol(:o), ((x, f(x)) for x in g)))
end
```

[`basis_at`](@ref) provides an *iterable* of the basis functions evaluated at a point:

```@repl linear_combination
collect(basis_at(basis, 0.3))
```

Importantly, since basis functions are calculated recursively, this is not indexable as a vector. It serves as a building block for [`linear_combination`](@ref).

## Domain and values outside it

You can use [`domain`](@ref) to query the domain of a transformation:

```@repl domain
using SpectralKit
b_strict = UnivariateBasis(Chebyshev(), Interior(), BoundedLinear(0.0, 3.0), 2)
domain(b_strict)
```

Domains support `extrema`:
```@repl domain
extrema(domain(b_strict))
```

The default behavior is to error for inputs outside the domain:

```@repl domain
linear_combination(b_strict, ones(dimension(b_strict)), 3.0001)
```

but this can be overridden:

```@repl domain
b_tolerant = UnivariateBasis(Chebyshev(), Interior(),
                             BoundedLinear(0.0, 3.0, NearestInDomain()), 2)
linear_combination(b_tolerant, ones(dimension(b_tolerant)), 3.0001)
```

Transformations that do not cover the entire real line have an option to specify what happens outside the domain. This is for **handling floating point error**: do not deliberately evaluate bases outside their domain.

## Dimensions without transformations

If you don't need grid values or linear combinations, you can specify an arbitrary value, such as `0` or `nothing`, in place of transformations. This allows the user to obtain the just *dimension* of the basis or the grid.

```@repl just-dimension
using SpectralKit
dimension(UnivariateBasis(Chebyshev(), Interior(), nothing, 2))
```

This also works with [Smolyak bases](@ref smolyak), and/or `length(grid(basis))`.

## Changing one aspect of a basis

[`UnivariateBasis`](@ref) and [`SmolyakBasis`](@ref) are just containers for their properties. This means that if you just want to change some aspects of a basis, you can use a library like [Accessors.jl](https://github.com/JuliaObjects/Accessors.jl):

```@repl
using SpectralKit, Accessors
basis = UnivariateBasis(Chebyshev(), Interior(), (BoundedLinear(1, 2), SemiInfRational()), 2)
basis2 = @set basis.level = 2
```
