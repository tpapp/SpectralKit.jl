#####
##### utilities
#####

####
#### infinite iteration
####

"""
$(SIGNATURES)

This package introduces its own API for infinite iteration. `_start(itr)` is not unlike
`iterate(itr)`, while `_next(itr, state) → x, state` is not unlike `iterate(itr,
state)`. Also see [`_eltype`](@ref).

The rationale is to ease the compilation burden by ruling out `Union` types (`nothing`)
and combinatorial explosition. Julia can cope with it `iterate`, but it causes problems
with `Enzyme`.
"""
_start(itr) = iterate(itr)::Tuple

"""
$(SIGNATURES) → x, state

Internal API for infinite iteration. See [`_start`](@ref).
"""
_next(itr, state) = iterate(itr, state)::Tuple

"""
$(SIGNATURES) → Type

Internal API for infinite iteration. See [`_start`](@ref).
"""
_eltype(::Type{T}) where T = eltype(T)

"Counting integers from 1."
struct Counting end

_start(::Counting) = 1, 1

_next(::Counting, state) = state + 1, state + 1

_eltype(::Type{Counting}) = Int

####
#### printing
####

const _SUPERSCRIPT_DIGITS = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']

const _SUBSCRIPT_DIGITS = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉']

"""
$(SIGNATURES)

Print a nonnegative number using `digits`, where `0` is indexed with `1`.
"""
function print_number(io::IO, DIGITS, k::Integer)
    @argcheck k ≥ 0
    if k < 10
        print(io, DIGITS[k + 1])
    else
        for d in reverse(digits(k))
            print(io, DIGITS[d + 1])
        end
    end
end

"""
Wrapper to `print` a nonnegative integer as superscript using Unicode.
"""
struct SuperScript
    i::Int
end

Base.print(io::IO, s::SuperScript) = print_number(io, _SUPERSCRIPT_DIGITS, s.i)

"""
Wrapper to `print` a nonnegative integer as subscript using Unicode.
"""
struct SubScript
    i::Int
end

Base.print(io::IO, s::SubScript) = print_number(io, _SUBSCRIPT_DIGITS, s.i)

# FIXME function below is currently unused, decide if we need to keep it after
# refactoring Base.show(::IO, ::∂Expansion)
# """
# $(SIGNATURES)

# Print notation for partial derivatives, where `d[i]` stands for ``∂ⁱ/∂x[d]ⁱ``.
# """
# function _print_partial_notation(io::IO, d)
#     if d ≡ ()
#         print(io, "value")
#     else
#         for (i, p) in enumerate(d)
#             print(io, "∂", SubScript(i), SuperScript(p))
#         end
#     end
# end

####
#### conversions
####

"""
$(SIGNATURES)

If `T <: NTuple{N}`, convert `v` into an `NTuple{N}`.

Used for ingesting `::AbstractVector` arguments in contexts where an `NTuple` or
`SVector` is preferred.
"""
function _ntuple_like(::Type{T}, v::AbstractVector) where {N,T<:NTuple{N}}
    @argcheck length(v) == N
    NTuple{N}(v)
end
