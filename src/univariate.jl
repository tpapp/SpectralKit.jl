####
#### Chebyshev polynomials on [-1,1]
####

export Chebyshev, Endpoints, Interior, UnivariateBasis

####
#### generic building blocks
####

"""
$(TYPEDEF)

The first `N` Chebyhev polynomials of the first kind, defined on `[-1,1]`.
"""
struct Chebyshev end

####
#### basis function iterator
####

struct ChebyshevIterator{T}
    x::T
end

function _start(itr::ChebyshevIterator{T}) where T
    (; x) = itr
    _one(T), (_one(T), x)
end

function _next(itr::ChebyshevIterator{T}, (fp, fpp)) where T
    (; x) = itr
    f = _sub(_mul(2, x, fp), fpp)
    f::T, (f, fp)
end

_eltype(::Type{<:ChebyshevIterator{T}}) where {T} = T

"""
$(SIGNATURES)

Helper function to calculate the extrema of the `N`th Chebyshev polynomial, indexed by
`1 ≤ i ≤ N` (not checked). Results are in `[-1,1]`. `N==1` (the constant) is special
cased to zero, for nesting.
"""
function _chebyshev_extremum(::Type{T}, i::Int, N::Int) where {T <: Real}
    if N == 1
        cospi(1/T(2))::T        # 0.0 as a practical fallback
    else
        cospi((N - i) ./ T(N - 1))::T
    end
end

####
#### kinds
####

"""
$(TYPEDEF)

Like [`Endpoints`](@ref), but with endpoints dropped.
"""
struct Interior end

"""
$(SIGNATURES)

Length of a univariate grid.
"""
function grid_length(::Chebyshev, ::Interior, level::Int)
    @argcheck level ≥ 0
    (1 << (level + 1)) - 1
end

"""
$(SIGNATURES)

Length of a single block, these are concatenated to form the grid.
"""
function block_length(::Chebyshev, ::Interior, level::Int)
    @argcheck level ≥ 0
    1 << level
end

"""
$(SIGNATURES)

Map `i` to an integer for calling [`_chebyshev_extremum_shuffle`](@ref). Interior
indices start from `2`, endpoint from `1`. Caller is responsible for making sure that
`i` is in the valid range `1:grid_length(Chebyshev(), kind, level)`, this is not checked.
"""
function _chebyshev_extremum_shuffle(kind::Interior, i::Int, level::Int)
    p = ndigits(i, base = 2) - 1    # trust constant folding fast path to top_set_bit
    remainder = i - (1 << p)
    start = 1 << (level - p)
    start + remainder * (start << 1) + 1
end

"""
$(TYPEDEF)

Chebyshev-Lobatto grid. The extrema of Chebyshev polynomials, including endpoints of `[-1,1]`.

!!! note
    For small dimensions may fall back to a grid that does not contain endpoints.
"""
struct Endpoints end

function grid_length(::Chebyshev, ::Endpoints, level::Int)
    @argcheck level ≥ 0
    (level ≤ 1 ? level * 2 : (1 << level)) + 1
end

function block_length(::Chebyshev, ::Endpoints, level::Int)
    @argcheck level ≥ 0
    level ≤ 1 ? (level + 1) : (1 << (level - 1))
end

function _chebyshev_extremum_shuffle(::Endpoints, i::Int, level::Int)
    if i > 3
        _chebyshev_extremum_shuffle(Interior(), i - 2, level - 1)
    elseif i == 1
        1 + 1 << (level - 1)
    else
        1 + (i - 2) * (1 << level)
    end
end

####
#### univariate bases
####

"""
Implementation of univariate bases. Not part of the API.
"""
struct UnivariateBasis{F,K,D} <: FunctionBasis
    family::F
    kind::K
    domain_transformation::D
    level::Int
    @doc """
    $(SIGNATURES)

    Univariate basis from `family`, using the given `kind`.

    `level` is an integer, starting from `0`, specifying the number of *blocks* used to
    build the grid.
    """
    function UnivariateBasis(family::F, kind::K, domain_transformation::D,
                             level::Int) where {F,K,D}
        @argcheck level ≥ 0
        new{F,K,D}(family, kind, domain_transformation, level)
    end
end

function Base.show(io::IO, basis::UnivariateBasis)
    (; family, kind, domain_transformation, level) = basis
    lead = "UnivariateBasis("
    next = ",\n" * ' '^length(lead)
    print(io, "UnivariateBasis(", family,
          next, kind, next, domain_transformation, next,
          level, ")    # dimension: ", dimension(basis))
end

domain(U::UnivariateBasis) = domain(U.domain_transformation)

dimension(U::UnivariateBasis) = grid_length(U.family, U.kind, U.level)

struct UnivariateBasisAt{I}
    infinite_itr::I
    N::Int
end

Base.eltype(::Type{UnivariateBasisAt{I}}) where I = _eltype(I)

Base.length(itr::UnivariateBasisAt) = itr.N

function Base.iterate(itr::UnivariateBasisAt, state = nothing)
    (; infinite_itr, N) = itr
    if state ≡ nothing
        x, inner_state = _start(infinite_itr)
        x, (1, inner_state)
    else
        i, inner_state = state
        if i < N
            x, inner_state′ = _next(infinite_itr, inner_state)
            x, (i + 1, inner_state′)
        else
            nothing
        end
    end
end

"""
$(SIGNATURES)

Return an infinite iterator for the univariate basis functions, using the protocol with
[`_start`](@ref), [`_next`](@ref), etc.
"""
function _univariate_basis_itr(family::Chebyshev, domain_transformation, x::Scalar)
    ChebyshevIterator(transform_to(PM1(), domain_transformation, x))
end

function basis_at(U::UnivariateBasis, x::Scalar)
    UnivariateBasisAt(_univariate_basis_itr(U.family, U.domain_transformation, x),
                      dimension(U))
end

@concrete struct ChebyshevGrid{T} <: AbstractVector{T}
    kind
    domain_transformation
    level::Int
    N::Int
    N̂::Int
end

Base.size(g::ChebyshevGrid) = (g.N, )

function Base.getindex(g::ChebyshevGrid{T}, i::Int) where T
    (; kind, domain_transformation, level, N̂) = g
    transform_from(PM1(), domain_transformation,
                   _chebyshev_extremum(T, _chebyshev_extremum_shuffle(kind, i, level), N̂))::T
end

function grid(::Type{T},
              U::UnivariateBasis{Chebyshev,
                                 <:Union{Endpoints,Interior}}) where {T <: AbstractFloat}
    (; family, kind, domain_transformation, level) = U
    N = grid_length(family, kind, level)
    ChebyshevGrid{T}(kind, domain_transformation, level, N,
                     kind ≡ Interior() ? N̂ = N + 2 : N)
end

function adjust_coefficients(θ1::AbstractVector,
                             U1::UnivariateBasis{F}, U2::UnivariateBasis{F}) where F
    d1 = dimension(U1)
    d2 = dimension(U2)
    @argcheck length(θ1) == d1 "coefficients are not compatible with the first basis"
    @argcheck(U1.domain_transformation == U2.domain_transformation,
              "incompatible domain transformations")
    if d2 ≤ d1                  # truncate
        θ1[1:d2]
    else                        # pad with zeros
        θ2 = zeros(d2)
        θ2[1:d1] .= θ1
        θ2
    end
end
