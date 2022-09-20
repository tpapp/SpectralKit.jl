#####
##### Generic API
#####

export is_function_basis, dimension, domain, basis_at, linear_combination, InteriorGrid,
    InteriorGrid2, EndpointGrid, grid, collocation_matrix, augment_coefficients,
    is_subset_basis

"""
$(TYPEDEF)

Abstract type for function families.

Not part of the API, just used internally for dispatch. See [`is_function_basis`](@ref).
"""
abstract type FunctionBasis end

Broadcast.broadcastable(basis::FunctionBasis) = Ref(basis)

"""
`$(FUNCTIONNAME)(F)`

`$(FUNCTIONNAME)(f::F)`

Test if the argument is a *function basis*, supporting the following interface:

- [`domain`](@ref) for querying the domain,

- [`dimension`](@ref) for the dimension,

- [`basis_at`](@ref) for function evaluation,

- [`grid`](@ref) to obtain collocation points.

[`linear_combination`](@ref) and [`collocation_matrix`](@ref) are also supported, building
on the above.

Can be used on both types (preferred) and values (for convenience).
"""
is_function_basis(T::Type) = false

is_function_basis(::Type{<:FunctionBasis}) = true

is_function_basis(f) = is_function_basis(typeof(f))

"""
`$(FUNCTIONNAME)(basis)`

The domain of a function basis. A tuple of numbers (of arbitrary type, but usually
`Float64`), or a tuple of domains by coordinate.
"""
function domain end

"""
`$(FUNCTIONNAME)(basis)`

Return the dimension of `basis`, a positive `Int`.
"""
function dimension end

"""
`$(FUNCTIONNAME)(basis, x)`

Return an iterable with known element type and length (`Base.HasEltype()`,
`Base.HasLength()`) of basis functions in `basis` evaluated at `x`.

Univariate bases operate on real numbers, while for multivariate bases,
`StaticArrays.SVector` is preferred for performance, though all `<:AbstractVector` types
should work.

Methods are type stable.

!!! note
    Consequences are undefined when evaluating outside the domain.
"""
function basis_at end

@inline function _linear_combination(basis, θ, x, _check)
    _check && @argcheck dimension(basis) == length(θ)
    mapreduce(_mul, _add, θ, basis_at(basis, x))
end

"""
$(SIGNATURES)

Evaluate the linear combination of ``∑ θₖ⋅fₖ(x)`` of function basis ``f₁, …`` at `x`, for
the given order.

The length of `θ` should equal `dimension(θ)`.
"""
linear_combination(basis, θ, x) = _linear_combination(basis, θ, x, true)

# FIXME define a nice Base.show method
struct LinearCombination{B,T}
    basis::B
    θ::T
    function LinearCombination(basis::B, θ::T) where {B,T}
        @argcheck dimension(basis) == length(θ)
        new{B,T}(basis, θ)
    end
end

(l::LinearCombination)(x) = _linear_combination(l.basis, l.θ, x, false)

"""
$(SIGNATURES)

Return a callable that calculates `linear_combination(basis, θ, x)` when called with `x`.
"""
linear_combination(basis, θ) = LinearCombination(basis, θ)

"""
$(TYPEDEF)

Abstract type for all grid specifications.
"""
abstract type AbstractGrid end

"""
$(TYPEDEF)

Grid with interior points (eg Gauss-Chebyshev).
"""
struct InteriorGrid <: AbstractGrid end

"""
$(TYPEDEF)

Grid that includes endpoints (eg Gauss-Lobatto).

!!! note
    For small dimensions may fall back to a grid that does not contain endpoints.
"""
struct EndpointGrid <: AbstractGrid end

"""
$(TYPEDEF)

Grid with interior points that results in smaller grids than `InteriorGrid` when nested.
Equivalent to an `EndpointGrid` with endpoints dropped.
"""
struct InteriorGrid2 <: AbstractGrid end

"""
$(SIGNATURES)

Return a gridpoint for collocation, with `1 ≤ i ≤ dimension(basis)`.

`T` is used *as a hint* for the element type of grid coordinates, and defaults to `Float64`.
The actual type can be broadened as required. Methods are type stable.

!!! note
    Not all grids have this method defined, especially if it is impractical. See
    [`grid`](@ref), which is part of the API, this function isn't.
"""
gridpoint(basis, i) = gridpoint(Float64, basis, i)

"""
`$(FUNCTIONNAME)([T], basis)`

Return a grid recommended for collocation, with `dimension(basis)` elements.

`T` is used *as a hint* for the element type of grid coordinates, and defaults to `Float64`.
The actual type can be broadened as required. Methods are type stable.
"""
grid(basis) = grid(Float64, basis)

function grid(::Type{T}, basis) where {T<:Real}
    map(i -> gridpoint(T, basis, i), 1:dimension(basis))
end

"""
$(SIGNATURES)

Convenience function to obtain a “collocation matrix” at points `x`, which is assumed to
have a concrete `eltype`. The default is `x = grid(basis)`, specialized methods may exist
for this when it makes sense.

The collocation matrix may not be an `AbstractMatrix`, all it needs to support is `C \\ y`
for compatible vectors `y = f.(x)`.

Methods are type stable.
"""
function collocation_matrix(basis, x = grid(basis))
    @argcheck isconcretetype(eltype(x))
    N = dimension(basis)
    M = length(x)
    C = Matrix{eltype(basis_at(basis, first(x)))}(undef, M, N)
    for (i, x) in enumerate(x)
        for (j, f) in enumerate(basis_at(basis, x))
            C[i, j] = f
        end
    end
    C
end

"""
`$(FUNCTIONNAME)(basis1, basis2, θ1)`

Return a set of coefficients `θ2` for `basis2` such that
```julia
linear_combination(basis1, θ1, x) == linear_combination(basis2, θ2, x)
```
for any `x` in the domain. In practice this means padding with zeros.

Throw a `ArgumentError` if the bases are incompatible with each other or `x`, or this is not
possible. Methods may not be defined for incompatible bases, compatibility between bases can
be checked with [`is_subset_basis`](@ref).
"""
function augment_coefficients end

"""
$(SIGNATURES)

Return a `Bool` indicating whether coefficients in `basis1` can be augmented to `basis2`
with [`augment_coefficients`](@ref).

!!! note
    `true` does not mean that coefficients from `basis1` can just be padded with zeros,
    since they may be in different positions. Always use [`augment_coefficients`](@ref).
"""
is_subset_basis(basis1::FunctionBasis, basis2::FunctionBasis) = false
