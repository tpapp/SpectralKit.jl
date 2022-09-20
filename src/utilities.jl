const _SUPERSCRIPT_DIGITS = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹']

"""
$(SIGNATURES)

Print a nonnegative integer with unicode superscript characters.
"""
function print_unicode_superscript(io::IO, k::Integer)
    @argcheck k > 0
    if k < 10
        print(io, _SUPERSCRIPT_DIGITS[k + 1])
    else
        for d in digits(l)
            print(io, _SUPERSCRIPT_DIGITS[d + 1])
        end
    end
end
