# SpectralKit

This is a very simple package for *building blocks* of spectral methods. Its intended audience is users who are familiar with the theory and practice of these methods, and prefer to assemble their code from modular building blocks.

The following provides an overview of the interface for univariate function families on various domains:
```jldoctest
julia> using SpectralKit

julia> F = ChebyshevInterval(0, 2.0)  # Chebyshev polynomial family on [0,2] — see others below
ChebyshevInterval(0.0, 2.0)

julia> is_function_family(F)          # tells us that we support the interface below
true

julia> domain_extrema(F)              # endpoints
(0.0, 2.0)

julia> roots(F, 5)                    # Gauss-Chebyshev grid
5-element Vector{Float64}:
 0.04894348370484636
 0.412214747707527
 1.0
 1.5877852522924731
 1.9510565162951536

julia> augmented_extrema(F, 5)        # Gauss-Lobatto grid
5-element Vector{Float64}:
 0.0
 0.2928932188134524
 1.0
 1.7071067811865475
 2.0

julia> basis_function(F, 0.41, Order(0), 3) # value of the 3rd (starting at 1!) polynomial at 0.41
-0.3037999999999994

julia> basis_function(F, 0.41, OrdersTo(1), 3) # value and the derivative
2-element StaticArrays.SVector{2, Float64} with indices SOneTo(2):
 -0.3037999999999994
 -2.3600000000000008

julia> basis_function(F, 0.25, Order(0), Val(4)) # Val{K} for an SVector up to K
4-element StaticArrays.SVector{4, Float64} with indices SOneTo(4):
  1.0
 -0.75
  0.125
  0.5625

julia> collect(Iterators.take(basis_function(F, 0.25, Order(0)), 4)) # iterator
4-element Vector{Float64}:
  1.0
 -0.75
  0.125
  0.5625
```

## Abstract interface for function families

```@docs
Order
OrdersTo
is_function_family
domain_extrema
basis_function
linear_combination
roots
augmented_extrema
```

## Specific function families

```@docs
Chebyshev
ChebyshevSemiInf
ChebyshevInf
ChebyshevInterval
```
