#####
##### Internal implementation for derivatives. See docs of [`Derivatives`](@ref).
#####

export 𝑑, ∂

####
#### Operations we support, with fallbacks defined below. This is deliberately kept
#### minimal, now all versions are defined for commutative operators.
####

_one(::Type{T}) where {T<:Real} = one(T)

_add(x::Real, y::Real) = x + y

_sub(x::Real, y::Real) = x - y

_mul(x::Real, y::Real) = x * y

_mul(x, y, z) = _mul(_mul(x, y), z)

####
#### Univariate expansions and derivatives
####


# """
# A small AD framework used *internally*, for calculating derivatives.

# Supports only the operations required by this module. The restriction is deliberate,
# should not be used for arithmetic operators outside this package.

# See [`∂`](@ref) for the exposed API.
# """
struct 𝑑Expansion{Dp1,T}
    "The function value and derivatives. `Dp1` is the degree of the last derivative + 1."
    coefficients::SVector{Dp1,T}
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
    function Partials(I::NTuple{N,Int}) where N
        @argcheck all(i -> i ≥ 0, I)
        @argcheck I ≡ () || last(I) ≠ 0
        new{N}(I)
    end
end

function Partials(I::Integer...)
    N = length(I)
    while N > 0 && I[N] == 0
        @show N
        N -= 1
    end
    Partials(ntuple(i -> Int(I[i]), N))
end

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

function Base.isless(p1::Partials{N1}, p2::Partials{N2}) where {N1, N2}
    _is_strict_subset(p1, p2) && return true
    _is_strict_subset(p2, p1) && return false
    p1.I < p2.I
end

Base.isequal(p1::Partials, p2::Partials) = p1.I == p2.I

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

function _partials_canonical_expansion(::Val{N}, partials) where N
    result = OrderedSet{NTuple{N,Int}}()
    function _plus1_cartesian_indices(p::Partials{M}) where M
        (; I) = p
        @argcheck M ≤ N
        CartesianIndices(ntuple(i -> i ≤ M ? I[i] + 1 : 1, Val(N)))
    end
    for p in partials
        for ι in _plus1_cartesian_indices(p)
            i = map(i -> i - 1, Tuple(ι))
            if !(i ∈ result)
                push!(result, i)
            end
        end
    end
    result
end

# struct ∂Derivatives{K,M,Ds}
#     function ∂Derivatives{K,M,Ds}() where {K,M,Ds}
#         @argcheck K isa Int && K ≥ 0
#         @argcheck _check_M_Ds_consistency(Val(N), Val(Ds)) ≡ Val(true)
#         new{K,M,Ds}()
#     end
#     function ∂Derivatives(dimension::Val{K}, degree::Val{m}) where {K,m}
#         @argcheck K isa Int && K ≥ 1
#         @argcheck m isa Int && m ≥ 0
#         m == 0 && return new{0,(),((),)}()
#         z = ntuple(_ -> 0, Val(K - 1))
#         M = (z..., m)
#         Ds = ntuple(i -> (z..., i - 1), Val(m + 1))
#         new{K,M,Ds}()
#     end
# end

# ∂(k::Val{K}, m::Val{M}) where {K,M} = ∂Derivatives(k, m)

# @inline ∂(dimension::Int, degree::Int = 1) = ∂(Val(dimension), Val(degree))

# @generated function Base.:∪(∂ds::∂Derivatives...)
#     _get_Ds(::∂Derivatives{K,N,Ds}) where {K,N,Ds} = Ds
#     M, Ds = _calculate_M_Ds_union(Iterators.flatten((_get_Ds(∂d) for ∂d in ∂ds)))
#     K = length(M)
#     quote
#         ∂Derivatives{$(K), $(M), $(Ds)}()
#     end
# end

# function Base.:^(::∂Derivatives{K,M,D}, ::Val{Y}) where {K,M,D,Y}
#     @argcheck Y isa Int && Y ≥ 0
#     (dhead..., dlast) = D
#     @argcheck(all(iszero, dhead...),
#               "The ^ operator can only be applied for ∂ derivatives along one dimension.")
#     ∂Derivatives(Val(K), Val(M * Y))
# end

# function _collapse_D_union!(D::Vector{_PARTIALS})
#     D = collect(D)
#     U = Vector{Pair{Int,Int}}()
#     while !isempty(D)
#         M = reduce(_partials_max, D)
#         k = findfirst(d -> d > 0, M)
#         k ≡ nothing && break    # we are done
#         m = M[k]
#         push!(U, k => m)
#         _d = (ntuple(_ -> 0, k - 1)..., m)
#         filter!(d -> !_partials_isless_or_eq(d, _d), D)
#     end
#     U
# end

# function Base.show(io::IO, ::∂Derivatives{K,M,D}) where {K,M,D}
#     print(io, "partial derivatives (at least $(K) dimensions), up to ")
#     join(io, ("∂($(k), $(m))" for (k, m) in _collapse_D_union!(collect(_PARTIALS, D))), " ∪ ")
# end

# struct ∂DerivativesInput{P<:∂Derivatives,N,T}
#     partial_derivatives::P
#     x::SVector{N,T}
#     function ∂DerivativesInput(∂derivatives::P,
#                                x::SVector{N,T}) where {K,P<:∂Derivatives{K},N,T}
#         @argcheck N ≤ K "Can't differentiate the $(K)th element of a $(N)-element input."
#         new{P,N,T}(∂derivatives, x)
#     end
# end

# (∂derivatives::∂Derivatives)(x::SVector) = ∂DerivativesInput(partial_derivatives, x)

# """
# See [`_lift`](@ref). Internal.
# """
# struct ∂DerivativesInputLifted{P<:∂Derivatives,L<:Tuple}
#     partial_derivatives::P
#     lifted_x::L
# end

# """
# $(SIGNATURES)

# Lift a partial derivative calculation into a tuple of `Derivatives`. Internal.
# """
# @generated function _lift(∂x::∂DerivativesInput{<:∂Derivatives{K,M,N}}) where {K,M,N}
#     _lifted_x = [:(∂(Val($(i ≤ K ? M[i] : 0)))(x[$(i)])) for i in 1:N]
#     quote
#         x = ∂x.x
#         lifted_x = ($(_lifted_x...),)
#         ∂PartialDerivativesInputLifted(∂x.∂derivatives, lifted_x)
#     end
# end

# struct ∂Expansion{P<:∂Derivatives,N,T}
#     ∂derivatives::P
#     coefficients::SVector{N,T}
#     function ∂Expansion(∂derivatives::∂Derivatives{K,M,D},
#                         coefficients::SVector{N,T}) where {K,M,D,N,T}
#         @argcheck length(D) == N
#         new{typeof(P),N,T}(∂derivatives, coefficients)
#     end
# end

# function Base.show(io::IO, expansion::∂Expansion{<:∂Derivatives{K,M,D}}) where {K,M,D}
#     (; coefficients) = expansion
#     print(io, "multivariate expansion:")
#     for (c, d) in enumerate(zip(coefficiends, D))
#         print(io, "\n")
#         _print_partial_notation(io, d)
#         print(io, " ", c)
#     end
# end

# """
# $(SIGNATURES)

# Conceptually equivalent to `prod(getindex.(sources, indices))`, which it returns when
# `kind` is `nothing`, a placeholder calculating any derivatives. Internal.
# """
# _product(kind::Nothing, sources, indices) = mapreduce(getindex, *, sources, indices)

# """
# $(SIGNATURES)

# Type that is returnedby [`_product`](@ref).
# """
# function _product_type(::Type{Nothing}, source_eltypes)
#     mapfoldl(eltype, promote_type, source_eltypes)
# end

# @generated function _product(∂derivatives::∂Derivatives{K,M,D},
#                              sources::NTuple{N},
#                              indices) where {K,M,D,N}
#     xs = [gensym(:x) for _ in 1:N]
#     assignments = [:($(xs[i]) = sources[indices[$(i)]]) for i in 1:N]
#     products = [begin
#                     ι = NTuple(i -> i ≤ K ? d[i] + 1 : 1, Val(N))
#                     mapreduce(i -> :($(xs[i])[$(ι[i])]),
#                               (a, b) -> :($(a) * $(b)),
#                               1:N)
#                 end for d in D]
#     quote
#         $(assignments...)
#         ∂Expansion(∂derivatives, SVector($(products)...))
#     end
# end

# function _product_type(::Type{∂Derivatives{M,L}}, source_eltypes) where {M,L}
#     T = _product_type(Nothing, map(eltype, source_eltypes))
#     N = length(fieldtypes(L))
#     ∂Output{N,T}
# end

# function _add(x::∂Expansion{P}, y::∂Expansion{P}) where P
#     ∂Expansion(x.∂derivatives, map(+, x.values, y.values))
# end

# _mul(x::Real, y::∂Expansion) = ∂Expansion(y.∂derivatives, map(y -> _mul(x, y), y.values))

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
