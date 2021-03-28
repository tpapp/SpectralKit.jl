#####
##### Generic API
#####

export is_function_family, domain, basis_at, linear_combination, InteriorGrid, EndpointGrid,
    grid

"""
$(TYPEDEF)

Abstract type for function families.

Not part of the API, just used internally for dispatch. See [`is_function_family`](@ref).
"""
abstract type FunctionFamily end

Broadcast.broadcastable(family::FunctionFamily) = Ref(family)

"""
`$(FUNCTIONNAME)(F)`

`$(FUNCTIONNAME)(f::F)`

Test if the argument is a *function family*, supporting the following interface:

- [`domain`](@ref) for querying the domain,

- [`basis_at`](@ref) for function evaluation,

- [`interior_grid`](@ref) and [`endpoint_grid`](@ref) to obtain collocation points.

Can be used on both types (preferred) and values (for convenience).
"""
is_function_family(::Type{Any}) = false

is_function_family(::Type{<:FunctionFamily}) = true

is_function_family(f) = is_function_family(typeof(f))

"""
`$(FUNCTIONNAME)(family)`

The domain of a function family. Can be an arbitrary object, but has to be constant.
"""
function domain end

"""
`$(FUNCTIONNAME)(family, x)`

Return an iterable with known element type and length (`Base.HasEltype()`,
`Base.HasLength()`) of basis functions in `family` evaluated at `x`.

The implementation is intended to be type stable.

!!! note
    Consequences are undefined when evaluating outside the domain.
"""
function basis_at end

"""
$(SIGNATURES)

Evaluate the linear combination of ``∑ θₖ⋅fₖ(x)`` of function family ``f₁, …`` at `x`, for
the given order.

The dimension is implicitly taken from `θ`.
"""
function linear_combination(family, x, θ)
    mapreduce(*, +, θ, basis_at(family, x))
end

"""
$(TYPEDEF)

Grid with interior points (eg Gauss-Chebyshev).
"""
struct InteriorGrid end

"""
$(TYPEDEF)

Grid that includes endpoints (eg Gauss-Lobatto).
"""
struct EndpointGrid end

"""
`$(FUNCTIONNAME)([T], family, kind)`

Return a grid the given `kind`, recommended for collocation.

`T` is used for the element type of grid coordinates, and defaults to `Float64`.
"""
grid(family, kind) = grid(Float64, family, kind)
