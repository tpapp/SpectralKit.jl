#####
##### Generic API
#####

export is_function_basis, dimension, basis_at, linear_combination, InteriorGrid,
    InteriorGrid2, EndpointGrid, grid, collocation_matrix, augment_coefficients,
    is_subset_basis, transformation

"""
$(TYPEDEF)

Abstract type for function families.

Not part of the API, just used internally for dispatch. See [`is_function_basis`](@ref).
"""
abstract type FunctionBasis end

Broadcast.broadcastable(basis::FunctionBasis) = Ref(basis)

abstract type UnivariateBasis <: FunctionBasis end

abstract type MultivariateBasis <: FunctionBasis end

"""
`$(FUNCTIONNAME)(::Type{F})`

`$(FUNCTIONNAME)(f::F)`

Test if the argument (value or type) is a *function basis*, supporting the following
interface:

- [`domain`](@ref) for querying the domain,

- [`dimension`](@ref) for the dimension,

- [`basis_at`](@ref) for function evaluation,

- [`grid`](@ref) to obtain collocation points.

- `length` and `getindex` for multivariate bases `(domain_kind(domain(basis)) ==
  :multivariate)`, getindex returns a compatible marginal basis

[`linear_combination`](@ref) and [`collocation_matrix`](@ref) are also supported, building
on the above.

Can be used on both types (preferred) and values (for convenience).
"""
is_function_basis(T::Type) = false

is_function_basis(::Type{<:FunctionBasis}) = true

is_function_basis(f) = is_function_basis(typeof(f))

"""
`$(FUNCTIONNAME)(basis)`

The domain of a function basis.

`$(FUNCTIONNAME)(transformation)`

The (co)domain of a transformation. The “other” domain (codomain, depending on the
mapping) is provided explicitly for transformations, and should be compatible with
the`domain` of the basis.

See [`domain_kind`](@ref) for the interface supported by domains.
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

Univariate bases operate on real numbers, while for multivariate bases, `Tuple`s or
`StaticArrays.SVector` are preferred for performance, though all `<:AbstractVector`
types should work.

Methods are type stable.

!!! note
    Consequences are undefined when evaluating outside the domain.
"""
function basis_at end

"""
$(SIGNATURES)

Helper function for iteration. Errors when arguments do not finish at the same time.
Internal.
"""
@inline _is_done(::Nothing, ::Nothing) = true
@inline _is_done(θI::Nothing, BI::Tuple) = throw(ArgumentError("not enough coefficients"))
@inline _is_done(θI::Tuple, BI::Nothing) = throw(ArgumentError("too many coefficients"))
@inline _is_done(::Tuple, ::Tuple) = false

"""
$(SIGNATURES)

Helper function for linear combinations of basis elements at `x`. Always checks that `θ`
and `basis` have compatible dimensions.

!!! NOTE
    `x` and `θ` can be anything that supports `_mul(θ[i], b[i])` and `_add` on the
    result of this.
"""
@inline function _linear_combination(basis, θ, x)
    # an implementation of mapreduce, to work around
    # https://github.com/JuliaLang/julia/issues/50735
    B = basis_at(basis, x)
    a = iterate(θ)
    b = iterate(B)
    @argcheck !_is_done(a, b)
    s = _mul(a[1], b[1])
    while true
        a = iterate(θ, a[2])
        b = iterate(B, b[2])
        _is_done(a, b) && return s
        s = _add(s, _mul(a[1], b[1]))
    end
end

"""
$(SIGNATURES)

Evaluate the linear combination of ``∑ θₖ⋅fₖ(x)`` of function basis ``f₁, …`` at `x`, for
the given order.

The length of `θ` should equal `dimension(θ)`.
"""
linear_combination(basis, θ, x) = _linear_combination(basis, θ, x)

# FIXME define a nice Base.show method
struct LinearCombination{B,C}
    basis::B
    θ::C
    function LinearCombination(basis::B, θ::C) where {B,C}
        @argcheck dimension(basis) == length(θ)
        new{B,C}(basis, θ)
    end
end

(l::LinearCombination)(x) = _linear_combination(l.basis, l.θ, x)

"""
$(SIGNATURES)

Return a callable that calculates `linear_combination(basis, θ, x)` when called with `x`.

You can use `linear_combination(basis, θ) ∘ transformation` for domain transformations,
though working with `basis ∘ transformation` may be preferred.
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
`$(FUNCTIONNAME)([T], basis)`

Return an iterator for the grid recommended for collocation, with `dimension(basis)`
elements.

`T` for the element type of grid coordinates, and defaults to `Float64`.
Methods are type stable.
"""
grid(basis) = grid(Float64, basis)

"""
$(SIGNATURES)

Convenience function to obtain a “collocation matrix” at points `x`, which is assumed to
have a concrete `eltype`. The default is `x = grid(basis)`, specialized methods may exist
for this when it makes sense.

The collocation matrix may not be an `AbstractMatrix`, all it needs to support is `C \\ y`
for compatible vectors `y = f.(x)`.

Methods are type stable. The elements of `x` can be derivatives, see [`𝑑`](@ref).
"""
function collocation_matrix(basis, x = grid(basis))
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

####
#### transformed basis
####

"""
Transform the domain of a basis.
"""
struct TransformedBasis{B,T} <: FunctionBasis
    parent::B
    transformation::T
    function TransformedBasis(parent::B, transformation::T) where {B,T}
        @argcheck domain_kind(domain(parent)) ≡ domain_kind(T)
        new{B,T}(parent, transformation)
    end
end

function Base.:(∘)(parent::FunctionBasis, transformation)
    TransformedBasis(parent, transformation)
end

Base.parent(basis::TransformedBasis) = basis.parent

transformation(basis::TransformedBasis) = basis.transformation

domain(basis::TransformedBasis) = domain(basis.transformation)

dimension(basis::TransformedBasis) = dimension(basis.parent)

function basis_at(basis::TransformedBasis, x)
    (; parent, transformation) = basis
    basis_at(parent, transform_to(domain(parent), transformation, x))
end

function grid(basis::TransformedBasis)
    (; parent, transformation) = basis
    d = domain(parent)
    Iterators.map(x -> transform_from(d, transformation, x), grid(parent))
end

function Base.:(∘)(linear_combination::LinearCombination, transformation)
    (; basis, θ) = linear_combination
    LinearCombination(basis ∘ transformation, θ)
end

Base.length(basis::TransformedBasis{<:MultivariateBasis}) = length(basis.parent)

function Base.getindex(basis::TransformedBasis{<:MultivariateBasis}, i::Int)
    (; parent, transformation) = basis
    TransformedBasis(parent[i], Tuple(transformation)[i])
end

function is_subset_basis(basis1::TransformedBasis, basis2::TransformedBasis)
    basis1.transformation ≡ basis2.transformation &&
        is_subset_basis(basis1.parent, basis2.parent)
end

function augment_coefficients(basis1::TransformedBasis, basis2::TransformedBasis, θ1)
    @argcheck is_subset_basis(basis1, basis2)
    augment_coefficients(basis1.parent, basis2.parent, θ1)
end

function transform_to(basis::FunctionBasis, transformation, x)
    transform_to(domain(basis), transformation, x)
end

function transform_from(basis::FunctionBasis, transformation, x)
    transform_from(domain(basis), transformation, x)
end
