#####
##### Internal implementation for derivatives.
#####

export 𝑑, ∂

####
#### Operations we support for calculating (untransformed) basis functions and their
#### products, with fallbacks defined below. This is deliberately kept minimal, not all
#### versions are defined for commutative operators.
####

_one(::Type{T}) where {T<:Real} = one(T)

_add(x::T, y::T) where {T <: Union{Real,SVector}} = x + y

_sub(x::Real, y::Real) = x - y

_mul(x::Union{Real,SVector}, y::Real) = x * y

_mul(x, y, z) = _mul(_mul(x, y), z)

####
#### Univariate expansions and derivatives
####

struct 𝑑Expansion{Dp1,T}
    "The function value and derivatives. `Dp1` is the degree of the last derivative + 1."
    coefficients::SVector{Dp1,T}
    @doc """
    $(SIGNATURES)

    Taylor expansion around a given value.

    Implements small AD framework used *internally*, for calculating derivatives, that
    supports only the operations required by this module. The restriction is deliberate,
    should not be used for arithmetic operators outside this package.

    See [`∂`](@ref) for the exposed API. This type supports `eltype` and `getindex`.
    """
    function 𝑑Expansion(coefficients::SVector{Dp1,T}) where {Dp1,T}
        new{Dp1,T}(coefficients)
    end
end

function Base.show(io::IO, x::𝑑Expansion)
    for (i, d) in enumerate(x.coefficients)
        i ≠ 1 && print(io, " + ")
        print(io, d)
        i ≥ 2 && print(io, "⋅Δ")
        i ≥ 3 && print(io, SuperScript(i - 1))
    end
end

Base.eltype(::Type{𝑑Expansion{Dp1,T}}) where {Dp1,T} = T

@inline Base.getindex(x::𝑑Expansion, i::Int) = x.coefficients[i + 1]

"""
$(SIGNATURES)

Requests the calculation of `D ≥ 0` derivatives within this package.

The preferred syntax is `𝑑^D`, which is type-stable.
"""
struct 𝑑Derivatives{D}
    function 𝑑Derivatives{D}() where D
        @argcheck D isa Int
        @argcheck D ≥ 0
        new{D}()
    end
end

Base.show(io::IO, ::𝑑Derivatives{D}) where D = print(io, "𝑑^$(D)")

"""
A callable that calculates the value and the derivative of the argument. Higher-order
derivatives can be obtained by using an exponent, or multiplication.

```jldoctest
julia> 𝑑(2.0)
2.0 + 1.0⋅Δ

julia> (𝑑^3)(2.0)
2.0 + 1.0⋅Δ + 0.0⋅Δ² + 0.0⋅Δ³
```

Note that non-literal exponentiation requires `^Val(y)`, for type stability.

See [`linear_combination`](@ref) for examples of evaluating derivatives of basis
functions and linear combinations.
"""
const 𝑑 = 𝑑Derivatives{1}()

Base.:*(::𝑑Derivatives{D1}, ::𝑑Derivatives{D2}) where {D1,D2} = 𝑑Derivatives{D1+D2}()

function Base.literal_pow(::typeof(^), ::𝑑Derivatives{D}, ::Val{Y}) where {D,Y}
    @argcheck Y isa Int && Y ≥ 0
    𝑑Derivatives{D*Y}()
end

# ^::Int deliberately unsupported
Base.:^(d::𝑑Derivatives, y::Val{Y}) where Y = Base.literal_pow(^, d, y)

function (::𝑑Derivatives{D})(x::T) where {D, T <: Real}
    𝑑Expansion(SVector(x, ntuple(i -> i == 1 ? one(T) : zero(T), Val(D))...))
end

function _one(::Type{𝑑Expansion{Dp1,T}}) where {Dp1,T}
    𝑑Expansion(SVector(ntuple(i -> i == 1 ? _one(T) : zero(T), Val(Dp1))))
end

function _add(x::𝑑Expansion{Dp1}, y::𝑑Expansion{Dp1}) where Dp1
    𝑑Expansion(map(_add, x.coefficients, y.coefficients))
end

function _sub(x::𝑑Expansion, y::𝑑Expansion)
    𝑑Expansion(map(_sub, x.coefficients, y.coefficients))
end

function _mul(x::Real, y::𝑑Expansion)
    𝑑Expansion(map(y -> _mul(x, y), y.coefficients))
end

@generated function _mul(x::𝑑Expansion{Dp1}, y::𝑑Expansion{Dp1}) where {Dp1}
    _sum_terms(k) = mapreduce(i -> :(_mul($(binomial(k, i)), xd[$(i + 1)], yd[$(k - i + 1)])),
                              (a, b) -> :(_add($(a), $(b))), 0:k)
    _derivatives(k) = mapfoldl(_sum_terms, (a, b) -> :($(a)..., $(b)), 0:(Dp1-1); init = ())
    quote
        xd = x.coefficients
        yd = y.coefficients
        𝑑Expansion(SVector($(_derivatives(Dp1))))
    end
end

"Types accepted as scalars in this package."
const Scalar = Union{Real,𝑑Expansion}

# FIXME incorporate into docs
# """
#     derivatives(x, ::Val(N) = Val(1))

# Obtain `N` derivatives (and the function value) at a scalar `x`. The `i`th derivative
# can be accessed with `[i]` from results, with `[0]` for the function value.

# # Important note about transformations

# Always use `derivatives` *before* a transformation for correct results. For example, for
# some transformation `t` and value `x` in the transformed domain,

# ```julia
# # right
# linear_combination(basis, θ, transform_to(domain(basis), t, derivatives(x)))
# # right (convenience form)
# (linear_combination(basis, θ) ∘ t)(derivatives(x))
# ```

# instead of

# ```julia
# # WRONG
# linear_combination(basis, θ, derivatives(transform_to(domain(basis), t, x)))
# ```

# For multivariate calculations, use the [`∂`](@ref) interface.

# # Example

# ```jldoctest
# julia> basis = Chebyshev(InteriorGrid(), 3)
# Chebyshev polynomials (1st kind), InteriorGrid(), dimension: 3

# julia> C = collect(basis_at(basis, derivatives(0.1)))
# 3-element Vector{SpectralKit.Derivatives{2, Float64}}:
#  1.0 + 0.0⋅Δ
#  0.1 + 1.0⋅Δ
#  -0.98 + 0.4⋅Δ

# julia> C[1][1]                         # 1st derivative of the linear term is 1
# 0.0
# ```
# """

####
#### partial derivatives API
####

struct Partials{N}
    I::NTuple{N,Int}
    """
    $(SIGNATURES)

    Partial derivatives up to given indices. Eg `Partials((1, 2))` would contain
    - the value,
    - first derivatives ``∂_1`, ``∂_2``,
    - cross derivatives ``∂_1 ∂_2``,
    - second derivative ``∂^2_2``

    This is just a building block used internally by `∂Derivatives`. The actual order in
    containers is determined by [`_partials_canonical_expansion`](@ref). See also
    [`_partials_minimal_representation`](@ref).

    This constructor enforces that the last index is non-zero. Use the other
    `Partials(I...)` constructor to strip trailing zeros. Note that the number indices
    just determines the *minimum* length for the multivariate arguments to be differentiated.
    """
    function Partials(I::NTuple{N,Int}) where N
        @argcheck all(i -> i ≥ 0, I)
        @argcheck I ≡ () || last(I) ≠ 0
        new{N}(I)
    end
end

function Partials(I::Integer...)
    N = length(I)
    while N > 0 && I[N] == 0
        N -= 1
    end
    Partials(ntuple(i -> Int(I[i]), N))
end

"""
$(SIGNATURES)

True iff the partial derivatives contained in the first argument is a *strict* subset of
those in the second.
"""
function _is_strict_subset(p1::Partials{N1}, p2::Partials{N2}) where {N1,N2}
    N1 > N2 && return false     # valid because derivatives are always positive
    I1 = p1.I
    I2 = p2.I
    strict = false
    for i in 1:min(N1, N2)
        I1[i] > I2[i] && return false
        strict |= I1[i] < I2[i]
    end
    strict || N1 < N2
end

"""
$(SIGNATURES)

Imposes a total ordering on `Partials` with the property that a strict subset implies an
order, but not necessarily vice versa. This determines the traversal and help with
eliminating nested specifications.
"""
function Base.isless(p1::Partials{N1}, p2::Partials{N2}) where {N1, N2}
    _is_strict_subset(p1, p2) && return true
    _is_strict_subset(p2, p1) && return false
    p1.I < p2.I
end

Base.isequal(p1::Partials, p2::Partials) = p1.I == p2.I

"""
$(SIGNATURES)

Collapse specification of partial derivatives (an iterable of `Partials`) so a canonical
“minimal” representation with respect to the total ordering. Eliminates nested
specifications and duplicates.

!!! NOTE
    Takes any iterable of `Partials`, returns a `Vector`. Allocates, for use in
    generated functions.
"""
function _partials_minimal_representation(partials)
    descending_partials = sort!(collect(Partials, partials); rev = true)
    minimal_partials = Vector{Partials}()
    for p in descending_partials
        if isempty(minimal_partials) || !_is_strict_subset(p, minimal_partials[end])
            push!(minimal_partials, p)
        end
    end
    minimal_partials
end

"""
$(SIGNATURES)

The ordering of partial derivatives contains for containers, for `N`-dimensional
arguments. Returns an iterable of `N`-tuples that contain integer indices of partial
derivatives, eg `(0, 1, 2)` for ``∂_2 ∂_3^2``.
"""
function _partials_canonical_expansion(::Val{N}, Ps) where N
    result = OrderedSet{NTuple{N,Int}}()
    function _plus1_cartesian_indices(p::Partials{M}) where M
        (; I) = p
        @argcheck M ≤ N
        CartesianIndices(ntuple(i -> i ≤ M ? I[i] + 1 : 1, Val(N)))
    end
    for p in Ps
        for ι in _plus1_cartesian_indices(p)
            i = map(i -> i - 1, Tuple(ι))
            if !(i ∈ result)
                push!(result, i)
            end
        end
    end
    result
end

"""
$(SIGNATURES)

Elementwise maximum of the iterable from `_partials_canonical_expansion`, an `N`-tuple
of nonnegative integers.
"""
function _partials_expansion_degrees(::Val{N}, partials) where N
    degrees = zero(MVector{N,Int})
    for P in partials
        for (j, i) in enumerate(P.I)
            degrees[j] = max(degrees[j], i)
        end
    end
    Tuple(degrees)
end

"""
$(SIGNATURES)

The smallest input dimension (length) a partial derivative specification can support.
"""
_partials_minimum_input_dimension(partials) = maximum(P -> length(P.I), partials)

"""
$(SIGNATURES)

Check if `Ps` is a minimal representation, ie a valid type parameter for `∂Derivatives`.
"""
function _is_minimal_representation(::Val{Ps}) where Ps
    _Ps = fieldtypes(Ps)
    _Ps isa Tuple{Vararg{Partials}} || return false
    _partials_minimal_representation(_Ps) == collect(_Ps)
end

struct ∂Derivatives{Ps}
    @doc """
    $(SIGNATURES)

    A callable that requests that the given partial derivatives of its argument are
    evaluated.

    The partial derivatives are encoded in the `Ps` as a `Tuple{...}` of `Partials`.
    They are checked to be “minimal”, see [`_is_minimal_representation`](@ref), except
    when `Ps` is a `Partials`, then it is wrapped and used as is.

    The API entry point is `∂`s, combined with `<<` and `∪`/`union`.
    """
    function ∂Derivatives{Ps}() where {Ps}
        if Ps isa Partials
            new{Tuple{Ps}}()
        else
            # FIXME check that this does not allocate
            @argcheck _is_minimal_representation(Val(Ps))
            new{Ps}()
        end
    end
end

"""
$(SIGNATURES)

Partial derivatives along the given coordinates.

!!! NOTE
    Partial derivatives are currently **experimental** and not heavily tested. API may
    change at any point without prior notice or deprecation.

The following are equivalent, and represent ``\\partial_1 \\partial^2_2``, ie the first
derivative along the first axis, and the second partial derivative along the second
axis.

```@jldoctest
julia> ∂(1, 2)
∂(1, 2)

julia> ∂((1, 2))
∂(1, 2)
```

Only the vararg form allows trailing zeros, which are stripped:
```@jldoctest
julia> ∂(1, 0)
∂(1)

julia> ∂((1, 0))
ERROR: ArgumentError: I ≡ () || last(I) ≠ 0 must hold.
```

Use the empty form for no derivatives:
```@jldoctest
julia> ∂()
∂()
```

Combine derivatives using `union` or `∪`:

```jldoctest
julia> ∂(1, 2) ∪ ∂(2, 1)
union(∂(2, 1), ∂(1, 2))
```
"""
∂(I::Tuple{Vararg{Int}}) = ∂Derivatives{Partials(I)}()

∂(I::Integer...) = ∂Derivatives{Partials(I...)}()

∂() = ∂Derivatives{Tuple{}}()

function Base.:<<(d::𝑑Derivatives{D}, ::Val{N}) where {D,N}
    @argcheck N isa Int && N ≥ 1
    @argcheck D ≥ 1
    ∂Derivatives{Partials(ntuple(i -> 0, Val(N - 1))..., D)}()
end

@generated function Base.:∪(∂ds::∂Derivatives...)
    _get_Ps(::Type{<:∂Derivatives{Ps}}) where {Ps} = [fieldtypes(Ps)...]
    Ps = mapreduce(_get_Ps, vcat, ∂ds)
    quote
        ∂Derivatives{Tuple{$(_partials_minimal_representation(Ps))...}}()
    end
end

function Base.show(io::IO, ::∂Derivatives{Ps}) where {Ps}
    _Ps = fieldtypes(Ps)
    _repr(P::Partials) = "∂($(join(P.I, ", ")))"
    if isempty(_Ps)
        print(io, "∂()")
    elseif length(_Ps) == 1
        print(io, _repr(_Ps[1]))
    else
        print(io, "union(")
        join(io, (_repr(P) for P in fieldtypes(Ps)), ", ")
        print(io, ")")
    end
end

@generated function _expand_coordinates(::∂Derivatives{Ps}, x::NTuple{N}) where {Ps,N}
    _Ps = fieldtypes(Ps)
    @argcheck N ≥ _partials_minimum_input_dimension(_Ps)
    M = _partials_expansion_degrees(Val(N), _Ps)
    x = [:(𝑑Derivatives{$(m)}()(x[$(j)])) for (j, m) in enumerate(M)]
    quote
        tuple($(x...))
    end
end

struct ∂CoordinateExpansion{D<:∂Derivatives,S<:Tuple}
    ∂D::D
    x::S
    function ∂CoordinateExpansion(∂D::D, x::S) where {D<:∂Derivatives,S<:Tuple}
        new{D,S}(∂D, x)
    end
end

(∂D::∂Derivatives)(x::Tuple) = ∂CoordinateExpansion(∂D, _expand_coordinates(∂D, x))

(∂D::∂Derivatives)(x::AbstractVector) = ∂D(Tuple(x))

struct ∂Expansion{D,N,T}
    ∂D::D
    coefficients::SVector{N,T}
    function ∂Expansion(∂D::D, coefficients::SVector{N,T}) where {D<:∂Derivatives,N,T}
        new{D,N,T}(∂D, coefficients)
    end
end

function _add(x::∂Expansion{D,N}, y::∂Expansion{D,N}) where {D,N}
    ∂Expansion(x.∂D, map(+, x.coefficients, y.coefficients))
end

function _mul(x::Real, y::∂Expansion)
    ∂Expansion(y.∂D, map(y -> _mul(x, y), y.coefficients))
end

# function Base.show(io::IO, expansion::∂Expansion{<:∂Derivatives{K,M,D}}) where {K,M,D}
#     (; coefficients) = expansion
#     print(io, "multivariate expansion:")
#     for (c, d) in enumerate(zip(coefficiends, D))
#         print(io, "\n")
#         _print_partial_notation(io, d)
#         print(io, " ", c)
#     end
# end

"""
$(SIGNATURES)

Conceptually equivalent to `prod(x)`, which it returns when `kind` is `nothing`, a
placeholder calculating any derivatives. Internal.
"""
_product(kind::Nothing, x::Tuple) = prod(x)

"""
$(SIGNATURES)

Type that is returnedby [`_product`](@ref).
"""
function _product_type(::Type{Nothing}, source_eltypes)
    mapfoldl(eltype, promote_type, source_eltypes)
end

@generated function _product(∂D::∂Derivatives{Ps}, x::NTuple{N,𝑑Expansion}) where {Ps,N}
    function _product(d)
        # FIXME could skip bounds checking if verified at the beginning
        mapreduce(i -> :(x[$i].coefficients[$(d[i]) + 1]), (a, b) -> :($(a) * $(b)), 1:N)
    end
    products = [_product(d) for d in _partials_canonical_expansion(Val(N), fieldtypes(Ps))]
    quote
        ∂Expansion(∂D, SVector($(products...)))
    end
end

function _product_type(::Type{D}, source_eltypes::NTuple{M}) where {Ps,D<:∂Derivatives{Ps},M}
    T = _product_type(Nothing, map(eltype, source_eltypes))
    N = length(_partials_canonical_expansion(Val(M), fieldtypes(Ps)))
    ∂Expansion{D,N,T}
end

#####
##### FIXME revise and move documentation below
#####


# """
# $(SIGNATURES)

# Partial derivative specification. The first argument is `Val(::Int)` or simply an `Int`
# (for convenience, using constant folding), determining the dimension of the argument.

# Subsequent arguments are indices of the input variable.

# ```jldoctest
# julia> ∂(3, (), (1, 1), (2, 3))
# partial derivatives
# [1] f
# [2] ∂²f/∂²x₁
# [3] ∂²f/∂x₂∂x₃
# ```
# """
# @inline function ∂(::Val{N}, partials...) where N
#     @argcheck N ≥ 1 "Needs at least one dimension."
#     @argcheck !isempty(partials) "Empty partial derivative specification."
#     lookups = map(p -> _partial_to_lookup(Val(N), p), partials)
#     M = ntuple(d -> maximum(l -> l[d], lookups), Val(N))
#     ∂Specification{M}(lookups)
# end

# @inline ∂(N::Integer, partials...) = ∂(Val(Int(N)), partials...)

# """
# Partial derivatives to be evaluated at some `x`. These need to be [`_lift`](@ref)ed,
# then combined with [`_product`](@ref) from bases. Internal, use `∂(specification, x)` to
# construct.
# """
# struct ∂Input{TS<:∂Specification,TX<:SVector}
#     ∂specification::TS
#     x::TX
#     function ∂Input(∂specification::TS, x::TX) where {M,N,TS<:∂Specification{M},TX<:SVector{N}}
#         @argcheck length(M) == N
#         new{TS,TX}(∂specification, x)
#     end
# end

# function Base.show(io::IO, ∂x::∂Input)
#     show(io, ∂x.∂specification)
#     print(io, "\nat ", ∂x.x)
# end

# """
# $(SIGNATURES)

# Input wrappert type for evaluating partial derivatives `∂specification` at `x`.

# ```jldoctest
# julia> using StaticArrays

# julia> s = ∂(Val(2), (), (1,), (2,), (1, 2))
# partial derivatives
# [1] f
# [2] ∂f/∂x₁
# [3] ∂f/∂x₂
# [4] ∂²f/∂x₁∂x₂

# julia> ∂(s, SVector(1, 2))
# partial derivatives
# [1] f
# [2] ∂f/∂x₁
# [3] ∂f/∂x₂
# [4] ∂²f/∂x₁∂x₂
# at [1, 2]
# ```
# """
# function ∂(∂specification::∂Specification{M}, x::Union{AbstractVector,Tuple}) where M
#     N = length(M)
#     ∂Input(∂specification, SVector{N}(x))
# end

# """
# $(SIGNATURES)

# Shorthand for `∂(x, ∂(Val(length(x)), partials...))`. Ideally needs an `SVector` or a
# `Tuple` so that size information can be obtained statically.
# """
# @inline function ∂(x::SVector{N}, partials...) where N
#     ∂specification = ∂(Val(N), partials...)
#     ∂Input(∂specification, x)
# end

# @inline ∂(x::Tuple, partials...) = ∂(SVector(x), partials...)


####
#### products (used by tensor / Smolyak bases)
####


# """
# Container for output of evaluating partial derivatives. Each corresponds to an
# specification in a [`∂Specification`](@ref). Can be indexed with integers, iterated, or
# converted to a `Tuple`.
# """
# struct ∂Output{N,T}
#     values::NTuple{N,T}
# end

# function Base.show(io::IO, ∂output::∂Output)
#     print(io, "SpectralKit.∂Output(")
#     join(io, ∂output.values, ", ")
#     print(io, ")")
# end

# @inline Base.Tuple(∂output::∂Output) = ∂output.values

# @inline Base.length(∂output::∂Output) = length(∂output.values)

# @inline Base.getindex(∂output::∂Output, i) = ∂output.values[i]

# @inline Base.iterate(∂output::∂Output, i...) = Base.iterate(∂output.values, i...)
