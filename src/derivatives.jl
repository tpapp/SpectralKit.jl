#####
##### Internal implementation for derivatives. See docs of [`Derivatives`](@ref).
#####

export derivatives, ∂

####
#### derivatives API
####

"""
A small AD framework used *internally*, for calculating derivatives.

Supports only the operations required by this module. The restriction is deliberate,
should not be used for arithmetic operators outside this package.

See [`derivatives`](@ref) for the exposed API.
"""
struct Derivatives{N,T}
    "The function value and derivatives."
    derivatives::NTuple{N,T}
    function Derivatives(derivatives::NTuple{N,T}) where {N,T}
        new{N,T}(derivatives)
    end
end

function Base.show(io::IO, x::Derivatives)
    for (i, d) in enumerate(x.derivatives)
        i ≠ 1 && print(io, " + ")
        print(io, d)
        i ≥ 2 && print(io, "⋅Δ")
        i ≥ 3 && print(io, SuperScript(i - 1))
    end
end

Base.eltype(::Type{Derivatives{N,T}}) where {N,T} = T

@inline Base.getindex(x::Derivatives, i::Int) = x.derivatives[i + 1]

"Types accepted as scalars in this package."
const Scalar = Union{Real,Derivatives}

"""
    derivatives(x, ::Val(N) = Val(1))

Obtain `N` derivatives (and the function value) at a scalar `x`. The `i`th derivative
can be accessed with `[i]` from results, with `[0]` for the function value.

# Important note about transformations

Always use `derivatives` *before* a transformation for correct results. For example, for
some transformation `t` and value `x` in the transformed domain,

```julia
# right
linear_combination(basis, θ, transform_to(domain(basis), t, derivatives(x)))
# right (convenience form)
(linear_combination(basis, θ) ∘ t)(derivatives(x))
```

instead of

```julia
# WRONG
linear_combination(basis, θ, derivatives(transform_to(domain(basis), t, x)))
```

For multivariate calculations, use the [`∂`](@ref) interface.

# Example

```jldoctest
julia> basis = Chebyshev(InteriorGrid(), 3)
Chebyshev polynomials (1st kind), InteriorGrid(), dimension: 3

julia> C = collect(basis_at(basis, derivatives(0.1)))
3-element Vector{SpectralKit.Derivatives{2, Float64}}:
 1.0 + 0.0⋅Δ
 0.1 + 1.0⋅Δ
 -0.98 + 0.4⋅Δ

julia> C[1][1]                         # 1st derivative of the linear term is 1
0.0
```
"""
function derivatives(x::T, ::Val{N} = Val(1)) where {N, T <: Real}
    Derivatives((x, ntuple(i -> i == 1 ? one(T) : zero(T), Val(N))...))
end

####
#### partial derivatives API
####

"""
A specification for partial derivatives. Each element of `lookup` is an N-dimensional
tuple, the elements of which determine the order of the derivative along that
coordinate, eg `(1, 2, 0)` means 1st derivative along coordinate 1, 2nd derivative along
coordinate 2.

The elementwise maximum is stored in `M`. It is a type parameter because it needs to be
available as such for `derivatives`. The implementation than calculates derivatives
along coordinates, and combines them according to `lookups`.

Internal.
"""
struct ∂Specification{M,L}
    lookups::L
    function ∂Specification{M}(lookups::L) where {M,L}
        @argcheck M isa Tuple{Vararg{Int}}
        @argcheck !isempty(lookups)
        @argcheck lookups isa Tuple{Vararg{typeof(M)}}
        new{M,L}(lookups)
    end
end

function Base.show(io::IO, ∂specification::∂Specification)
    (; lookups) = ∂specification
    print(io, "partial derivatives")
    for (j, lookup) in enumerate(lookups)
        print(io, "\n[$j] ")
        if all(iszero, lookup)
            print(io, "f")
        else
            s = sum(lookup)
            print(io, "∂")
            s ≠ 1 && print(io, SuperScript(s))
            print(io, "f/")
            for (i, l) in enumerate(lookup)
                l == 0 && continue # don't print ∂⁰
                print(io, "∂")
                if l ≠ 1
                    print(io, SuperScript(l))
                end
                print(io, "x", SubScript(i))
            end
        end
    end
end

"""
$(SIGNATURES)

Convert a partial specification to a lookup, for `N`-dimensional arguments. Eg

```jldoctest
julia> SpectralKit._partial_to_lookup(Val(3), (1, 3, 1))
(2, 0, 1)
```
"""
function _partial_to_lookup(::Val{N}, partial::Tuple{Vararg{Int}}) where N
    @argcheck all(p -> 0 < p ≤ N, partial)
    ntuple(d -> sum(p -> p == d, partial; init = 0), Val(N))
end

"""
$(SIGNATURES)

Partial derivative specification. The first argument is `Val(::Int)` or simply an `Int`
(for convenience, using constant folding), determining the dimension of the argument.

Subsequent arguments are indices of the input variable.

```jldoctest
julia> ∂(3, (), (1, 1), (2, 3))
partial derivatives
[1] f
[2] ∂²f/∂²x₁
[3] ∂²f/∂x₂∂x₃
```
"""
@inline function ∂(::Val{N}, partials...) where N
    @argcheck N ≥ 1 "Needs at least one dimension."
    @argcheck !isempty(partials) "Empty partial derivative specification."
    lookups = map(p -> _partial_to_lookup(Val(N), p), partials)
    M = ntuple(d -> maximum(l -> l[d], lookups), Val(N))
    ∂Specification{M}(lookups)
end

@inline ∂(N::Integer, partials...) = ∂(Val(Int(N)), partials...)

"""
Partial derivatives to be evaluated at some `x`. These need to be [`_lift`](@ref)ed,
then combined with [`_product`](@ref) from bases. Internal, use `∂(specification, x)` to
construct.
"""
struct ∂Input{TS<:∂Specification,TX<:SVector}
    ∂specification::TS
    x::TX
    function ∂Input(∂specification::TS, x::TX) where {M,N,TS<:∂Specification{M},TX<:SVector{N}}
        @argcheck length(M) == N
        new{TS,TX}(∂specification, x)
    end
end

function Base.show(io::IO, ∂x::∂Input)
    show(io, ∂x.∂specification)
    print(io, "\nat ", ∂x.x)
end

"""
$(SIGNATURES)

Input wrappert type for evaluating partial derivatives `∂specification` at `x`.

```jldoctest
julia> using StaticArrays

julia> s = ∂(Val(2), (), (1,), (2,), (1, 2))
partial derivatives
[1] f
[2] ∂f/∂x₁
[3] ∂f/∂x₂
[4] ∂²f/∂x₁∂x₂

julia> ∂(s, SVector(1, 2))
partial derivatives
[1] f
[2] ∂f/∂x₁
[3] ∂f/∂x₂
[4] ∂²f/∂x₁∂x₂
at [1, 2]
```
"""
function ∂(∂specification::∂Specification{M}, x::Union{AbstractVector,Tuple}) where M
    N = length(M)
    ∂Input(∂specification, SVector{N}(x))
end

"""
$(SIGNATURES)

Shorthand for `∂(x, ∂(Val(length(x)), partials...))`. Ideally needs an `SVector` or a
`Tuple` so that size information can be obtained statically.
"""
@inline function ∂(x::SVector{N}, partials...) where N
    ∂specification = ∂(Val(N), partials...)
    ∂Input(∂specification, x)
end

@inline ∂(x::Tuple, partials...) = ∂(SVector(x), partials...)

"""
See [`_lift`](@ref). Internal.
"""
struct ∂InputLifted{TS<:∂Specification,TL<:Tuple}
    ∂specification::TS
    lifted_x::TL
end

"""
$(SIGNATURES)

Lift a partial derivative calculation into a tuple of `Derivatives`. Internal.
"""
@generated function _lift(∂x::∂Input{<:∂Specification{M}}) where M
    _lifted_x = [:(derivatives(x[$(i)], Val($(m)))) for (i, m) in enumerate(M)]
    quote
        x = ∂x.x
        lifted_x = ($(_lifted_x...),)
        ∂InputLifted(∂x.∂specification, lifted_x)
    end
end

####
#### products (used by tensor / Smolyak bases)
####

"""
$(SIGNATURES)

Conceptually equivalent to `prod(getindex.(sources, indices))`, which it returns when
`kind` is `nothing`, a placeholder calculating any derivatives. Internal.
"""
_product(kind::Nothing, sources, indices) = mapreduce(getindex, *, sources, indices)

"""
$(SIGNATURES)

Type that is returnedby [`_product`](@ref).
"""
function _product_type(::Type{Nothing}, source_eltypes)
    mapfoldl(eltype, promote_type, source_eltypes)
end

"""
Container for output of evaluating partial derivatives. Each corresponds to an
specification in a [`∂Specification`](@ref). Can be indexed with integers, or converted
to a `Tuple`.
"""
struct ∂Output{N,T}
    values::NTuple{N,T}
end

function Base.show(io::IO, ∂output::∂Output)
    print(io, "SpectralKit.∂Output(")
    join(io, ∂output.values, ", ")
    print(io, ")")
end

@inline Base.Tuple(∂output::∂Output) = ∂output.values

@inline Base.length(∂output::∂Output) = length(∂output.values)

@inline Base.getindex(∂output::∂Output, i) = ∂output.values[i]

function _product(∂specification::∂Specification, sources, indices)
    (; lookups) = ∂specification
    p = map(lookups) do lookup
        mapreduce((l, s, i) -> s[i][l], *, lookup, sources, indices)
    end
    ∂Output(p)
end

function _product_type(::Type{∂Specification{M,L}}, source_eltypes) where {M,L}
    T = _product_type(Nothing, map(eltype, source_eltypes))
    N = length(fieldtypes(L))
    ∂Output{N,T}
end

####
#### operations we support
####
#### This is deliberately kept minimal, now all versions are defined for commutative
#### operators.
####

_one(::Type{T}) where {T<:Real} = one(T)

function _one(::Type{Derivatives{N,T}}) where {N,T}
    Derivatives(ntuple(i -> i == 1 ? _one(T) : zero(T), Val(N)))
end

_add(x::Real, y::Real) = x + y

function _add(x::Derivatives, y::Derivatives)
    Derivatives(map(_add, x.derivatives, y.derivatives))
end

function _add(x::∂Output, y::∂Output)
    ∂Output(map(+, x.values, y.values))
end

_sub(x::Real, y::Real) = x - y

function _sub(x::Derivatives, y::Derivatives)
    Derivatives(map(_sub, x.derivatives, y.derivatives))
end

_mul(x::Real, y::Real) = x * y

_mul(x, y, z) = _mul(_mul(x, y), z)

function _mul(x::Real, y::Derivatives)
    Derivatives(map(y -> _mul(x, y), y.derivatives))
end

_mul(x::Real, y::∂Output) = ∂Output(map(y -> _mul(x, y), y.values))

@generated function _mul(x::Derivatives{N}, y::Derivatives{N}) where {N}
    _sum_terms(k) = mapreduce(i -> :(_mul($(binomial(k, i)), xd[$(i + 1)], yd[$(k - i + 1)])),
                              (a, b) -> :(_add($(a), $(b))), 0:k)
    _derivatives(k) = mapfoldl(_sum_terms, (a, b) -> :($(a)..., $(b)), 0:(N-1); init = ())
    quote
        xd = x.derivatives
        yd = y.derivatives
        Derivatives($(_derivatives(N)))
    end
end
