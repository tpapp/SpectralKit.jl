#####
##### Internal implementation for derivatives. See docs of [`Derivatives`](@ref).
#####

export derivatives

"""
A small AD framework used *internally*, for calculating derivatives.

Supports only the operations required by this module. The restriction is deliberate, should
not be used for arithmetic operators outside this package.

See [`derivatives`](@ref).
"""
struct Derivatives{I,N,T}
    "The function value and derivatives."
    derivatives::NTuple{N,T}
    function Derivatives{I}(derivatives::NTuple{N,T}) where {I,N,T}
        @argcheck I isa Int
        @argcheck I ≥ 0
        new{I,N,T}(derivatives)
    end
end

@inline Base.getindex(x::Derivatives, i::Int) = x.derivatives[i + 1]

"Types accepted as scalars in this package."
const Scalar = Union{Real,Derivatives}

"""
$(SIGNATURES)

The highest `I` in the arguments `Derivatives{I}`, wrapped in a `Val`. `J` gives the lower
bound. Internal.
"""
_highest_tag(::Val{J}, x::Real, xs...) where {J} = _highest_tag(Val(J), xs...)

function _highest_tag(::Val{J}, x::Derivatives{I}, xs...) where {J,I}
    _highest_tag(Val(max(I,J)), xs...)
end

_highest_tag(::Val{J}) where {J} = Val(J)

"""
$(SIGNATURES)

Map `xs` to increasing tags when `0`, starting at `J`.
"""
function _increasing_tags(::Val{J}, acc, x::Derivatives{I}, xs...) where {J,I}
    if I == 0
        _increasing_tags(Val(J + 1), (acc..., Derivatives{J+1}(x.derivatives)), xs...)
    else
        _increasing_tags(Val(J), (acc..., x), xs...)
    end
end

function _increasing_tags(::Val{J}, acc, x::Real, xs...) where {J}
    _increasing_tags(Val(J), (acc..., x), xs...)
end

_increasing_tags(::Val{J}, acc) where {J} = acc

"""
$(SIGNATURES)

Replace zero tags with increasing numbers from left to right, starting after the highest tag
in the argument.
"""
function replace_zero_tags(xs::Tuple)
    _increasing_tags(_highest_tag(Val(0), xs...), (), xs...)
end

"""
    derivatives(::Val(I) = Val(0), x, ::Val(N) = Val(1))

Obtain `N` derivatives (and the function value) at a scalar `x`. The `i`th derivative can be
accessed with `[i]` from results, with `[0]` for the function value.

`I` is an integer “tag” for determining the nesting order. Lower `I` always end up outside
when nested. When the defaults are used in multiple coordinates, increasing numbers replace
zeros from left to right, starting after the highest explicitly assigned tag.

Consequently, for most applications, you only need to specify tags if you want a different
nesting than left-to-right.

# Important note about transformations

Always use `derivatives` *before* a transformation for correct results. For example, for some
transformation `I` and value `x` in the transformed domain,

```julia
basis_at(basis2, to_pm1(I, derivatives(x))) # right
```

instead of

```julia
basis_at(basis2, derivatives(to_pm1(I, x))) # WRONG
```

# Univariate example

```jldoctest
julia> basis = Chebyshev(InteriorGrid(), 3)
Chebyshev polynomials (1st kind), InteriorGrid(), dimension: 3

julia> C = collect(basis_at(basis, derivatives(0.1)))
3-element Vector{SpectralKit.Derivatives{0, 2, Float64}}:
 SpectralKit.Derivatives{0, 2, Float64}((1.0, 0.0))
 SpectralKit.Derivatives{0, 2, Float64}((0.1, 1.0))
 SpectralKit.Derivatives{0, 2, Float64}((-0.98, 0.4))

julia> C[1][1]                         # 1st derivative of the linear term is 1
0.0
```

# Multivariate example

```jldoctest
julia> basis = smolyak_basis(Chebyshev, InteriorGrid(), SmolyakParameters(2), 2)
Sparse multivariate basis on ℝ²
  Smolyak indexing, ∑bᵢ ≤ 2, all bᵢ ≤ 2, dimension 21
  using Chebyshev polynomials (1st kind), InteriorGrid(), dimension: 9

julia> C = collect(basis_at(basis, (derivatives(0.1), derivatives(0.2, Val(2)))));

julia> C[14][1][2]                  # ∂/∂x₁ ∂/∂x₂² of the 14th basis function at x
4.0
```
"""
function derivatives(::Val{I}, x::T, ::Val{N} = Val(1)) where {I, N, T <: Real}
    Derivatives{I}((x, ntuple(i -> i == 1 ? one(T) : zero(T), Val(N))...))
end

derivatives(x::Real, ::Val{N} = Val(1)) where {N} = derivatives(Val(0), x, Val(N))

####
#### operations we support
####
#### This is deliberately kept minimal, now all versions are defined for commutative
#### operators.
####

_zero(::Type{T}) where {T<:Real} = zero(T)

_zero(x::T) where T = _zero(T)

_one(::Type{T}) where {T<:Real} = one(T)

_one(x::T) where T = _one(T)

_add(x::Real, y::Real) = x + y

_sub(x::Real, y::Real) = x - y

_mul(x::Real, y::Real) = x * y

_mul(x, y, z) = _mul(_mul(x, y), z)

_div(x::Real, y::Real) = x / y

function _one(::Type{Derivatives{I,N,T}}) where {I,N,T}
    Derivatives{I}(ntuple(i -> i == 1 ? _one(T) : _zero(T), Val(N)))
end

function _add(x::Derivatives{I}, y::Derivatives{I}) where {I}
    Derivatives{I}(map(_add, x.derivatives, y.derivatives))
end

function _sub(x::Derivatives{I}, y::Real) where {I}
    x1, xrest... = x.derivatives
    Derivatives{I}((x1 - y, xrest...))
end

function _sub(x::Derivatives{I}, y::Derivatives{I}) where {I}
    Derivatives{I}(map(_sub, x.derivatives, y.derivatives))
end

function _mul(x::Real, y::Derivatives{I}) where {I}
    Derivatives{I}(map(y -> _mul(x, y), y.derivatives))
end

function _div(x::Derivatives{I}, y::Real) where {I}
    Derivatives{I}(map(x -> _div(x, y), x.derivatives))
end

@generated function _mul(x::Derivatives{I,N}, y::Derivatives{I,N}) where {I,N}
    _sum_terms(k) = mapreduce(i -> :(_mul($(binomial(k, i)), xd[$(i + 1)], yd[$(k - i + 1)])),
                              (a, b) -> :(_add($(a), $(b))), 0:k)
    _derivatives(k) = mapfoldl(_sum_terms, (a, b) -> :($(a)..., $(b)), 0:(N-1); init = ())
    quote
        xd = x.derivatives
        yd = y.derivatives
        Derivatives{$(I)}($(_derivatives(N)))
    end
end

function _nesting_mul(x::Derivatives{I}, y::Derivatives) where {I}
    Derivatives{I}(map(x -> _mul(x, y), x.derivatives))
end

function _mul(x::Derivatives{I1},y::Derivatives{I2}) where {I1,I2}
    @argcheck I1 ≠ I2
    @argcheck I1 > 0 && I2 > 0 "Can't nest derivatives with zero tags."
    if I1 < I2
        _nesting_mul(x, y)
    else
        _nesting_mul(y, x)
    end
end

function _mul_type(::Type{T1}, ::Type{T2}) where {T1<:Real,T2<:Real}
    promote_type(T1, T2)
end

function _mul_type(::Type{Derivatives{I1,N1,T1}},
                   ::Type{Derivatives{I2,N2,T2}}) where {I1,N1,T1,I2,N2,T2}
    if I1 < I2
        Derivatives{I1,N1,Derivatives{I2,N2,_mul_type(T1, T2)}}
    else
        Derivatives{I2,N2,Derivatives{I1,N1,_mul_type(T1, T2)}}
    end
end

function _mul_type(::Type{Derivatives{I1,N1,T1}},
                   ::Type{T2}) where {I1,N1,T1,T2<:Real}
    Derivatives{I1,N1,_mul_type(T1, T2)}
end

function _mul_type(S1::Type{<:Real}, S2::Type{<:Derivatives})
    _mul_type(S2, S1)
end
