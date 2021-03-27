#####
##### Generic API
#####

struct Order{D}
    function Order{D}() where D
        @argcheck D isa Integer && D ≥ 0
        new{D}()
    end
end

Broadcast.broadcastable(o::Order) = Ref(o)

"""
$(SIGNATURES)

(Evaluate) derivative `D`, ≥ 0.
"""
@inline Order(D::Integer) = Order{Int(D)}()

struct OrdersTo{D}
    function OrdersTo{D}() where D
        @argcheck D isa Integer && D ≥ 0
        new{D}()
    end
end

Broadcast.broadcastable(o::OrdersTo) = Ref(o)

"""
$(SIGNATURES)

(Evaluate) derivatives `0`, …, `D`.
"""
@inline OrdersTo(D::Integer) = OrdersTo{Int(D)}()

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

- [`domain_extrema`](@ref) for querying the domain,

- [`basis_function`](@ref) for function evaluation,

- [`roots`](@ref) and [`augmented_extrema`](@ref) to obtain collocation points.

Can be used on both types (preferred) and values (for convenience).
"""
is_function_family(::Type{Any}) = false

is_function_family(::Type{<:FunctionFamily}) = true

is_function_family(f) = is_function_family(typeof(f))

"""
`$(FUNCTIONNAME)(family)`

Return the extrema of the domain of the given `family` as a tuple.

Type can be arbitrary, but guaranteed to be the same for both endpoints, and type stable.
"""
function domain_extrema end

"""
`$(FUNCTIONNAME)(family, x, order, [k])`

Evaluate basis functions of `family` at `x`, up to the given `order`.

`k` can be one of the following:

- `k::Int ≥ 1` evaluates the `k`th (starting from 1) function in `family` at `x`.

- `k::Val{K}()` returns the first `K` function values in the family as an `SVector{K}`.

- when `k` is not provided, the function returns a *iterable* for all basis functions.

`order` determines the derivatives:

- `Order(d)` returns the `d`th derivative as a scalar, starting from `0` (for the function
  value)

- `OrdersTo(d)` returns the derivatives up to `d`, starting from the function value, as an
  `SVector`.

The implementation is intended to be type stable.

!!! note
    Consequences are undefined for evaluating outside the domain.

## Note about indexing

Most texts index polynomial families with `n = 0, 1, …`. Following the Julia array indexing
convention, this package uses `k = 1, 2, …`. Some code may use `n = k - 1` internally for
easier comparison with well-known formulas.
"""
function basis_function end

"""
$(SIGNATURES)

Evaluate the linear combination of ``∑ θₖ⋅fₖ(x)`` of function family ``f₁, …`` at `x`, for
the given order.

The dimension is implicitly taken from `θ`.
"""
function linear_combination(family, x, order, θ)
    mapreduce(*, +, θ, basis_function(family, x, order))
end

"""
`$(FUNCTIONNAME)([T], family, N)`

Return the roots of the `K = N + 1`th function in `family`, as a vector of `N` numbers with
element type `T` (default `Float64`).

In the context of collocation, this is also known as the “Gauss-Chebyshev” grid.

Order is monotone, but not guaranteed to be increasing.
"""
roots(family, N::Integer) = roots(Float64, family, N)

"""
`$(FUNCTIONNAME)([T], family, N)`

Return the augmented extrema (extrema + boundary values) of the `N`th function in `family`,
as a vector of `N` numbers with element type `T` (default `Float64`).

In the context of collocation, this is also known as the “Gauss-Lobatto” grid.

Order is monotone, but not guaranteed to be increasing.
"""
function augmented_extrema(family::FunctionFamily, N::Integer)
    augmented_extrema(Float64, family, N)
end
