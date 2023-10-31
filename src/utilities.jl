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
