#####
##### Generic API
#####

export is_function_basis, dimension, domain, basis_at, linear_combination,
    InteriorGrid, EndpointGrid, grid, collocation_matrix

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
`Float64`), or a tuple the latter.
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

Methods are type stable.

!!! note
    Consequences are undefined when evaluating outside the domain.
"""
function basis_at end

"""
$(SIGNATURES)

Evaluate the linear combination of ``∑ θₖ⋅fₖ(x)`` of function basis ``f₁, …`` at `x`, for
the given order.

The length of `θ` should equal `dimension(θ)`.
"""
function linear_combination(basis, θ, x)
    mapreduce(*, +, θ, basis_at(basis, x))
end

# FIXME define a nice Base.show method
struct LinearCombination{B,T}
    basis::B
    θ::T
end

(l::LinearCombination)(x) = linear_combination(l.basis, l.θ, x)

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

Base.show(io::IO, ::InteriorGrid) = print(io, "interior grid")

"""
$(TYPEDEF)

Grid that includes endpoints (eg Gauss-Lobatto).
"""
struct EndpointGrid <: AbstractGrid end

Base.show(io::IO, ::EndpointGrid) = print(io, "grid w/ endpoints")

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

Convenience function to obtain a collocation matrix at gridpoints `x`, which is assumed to
have a concrete `eltype`.

Methods are type stable.
"""
function collocation_matrix(basis, x)
    @argcheck isconcretetype(eltype(x))
    N = dimension(basis)
    C = Matrix{eltype(basis_at(basis, first(x)))}(undef, N, N)
    for i in 1:N
        foreach(((j, f),) -> C[i, j] = f, enumerate(basis_at(basis, x[i])))
    end
    C
end
