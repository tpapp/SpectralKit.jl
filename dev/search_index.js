var documenterSearchIndex = {"docs":
[{"location":"#SpectralKit-1","page":"SpectralKit","title":"SpectralKit","text":"","category":"section"},{"location":"#Abstract-interface-for-function-families-1","page":"SpectralKit","title":"Abstract interface for function families","text":"","category":"section"},{"location":"#","page":"SpectralKit","title":"SpectralKit","text":"is_function_family\ndomain_extrema\nevaluate\nroots\naugmented_extrema","category":"page"},{"location":"#SpectralKit.is_function_family","page":"SpectralKit","title":"SpectralKit.is_function_family","text":"is_function_family(F)\n\nis_function_family(f::F)\n\nTest if the argument is a function family, supporting the following interface:\n\ndomain_extrema for querying the domain,\nevaluate for function evaluation,\nroots and augmented_extrema to obtain collocation points.\n\nCan be used on both types (preferred) and values (for convenience).\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.domain_extrema","page":"SpectralKit","title":"SpectralKit.domain_extrema","text":"domain_extrema(family)\n\nReturn the extrema of the domain of the given family as a tuple.\n\nType can be arbitrary, but guaranteed to be the same for both endpoints, and type stable.\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.evaluate","page":"SpectralKit","title":"SpectralKit.evaluate","text":"evaluate(family, k, x, order)\n\nEvaluate the kth (starting from 1) function in family at x.\n\norder determines the derivatives:\n\nVal(d) returns the dth derivative, starting from 0 (for the function value)\nVal(0:d) returns the derivatives up to d, starting from the function value, as  multiple values (ie a tuple).\n\nThe implementation is intended to be type stable.\n\nnote: Note\nConsequences are undefined for evaluating outside the domain.\n\nNote about indexing\n\nMost texts index polynomial families with n = 0, 1, …. Following the Julia array indexing convention, this package uses k = 1, 2, …. Some code may use n = k - 1 internally for easier comparison with well-known formulas.\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.roots","page":"SpectralKit","title":"SpectralKit.roots","text":"roots([T], family, N)\n\nReturn the roots of the K = N + 1th function in family, as a vector of N numbers with element type T (default Float64).\n\nIn the context of collocation, this is also known as the “Gauss-Chebyshev” grid.\n\nOrder is monotone, but not guaranteed to be increasing.\n\n\n\n\n\n","category":"function"},{"location":"#SpectralKit.augmented_extrema","page":"SpectralKit","title":"SpectralKit.augmented_extrema","text":"augmented_extrema([T], family, N)\n\nReturn the augmented extrema (extrema + boundary values) of the Nth function in family, as a vector of N numbers with element type T (default Float64).\n\nIn the context of collocation, this is also known as the “Gauss-Lobatto” grid.\n\nOrder is monotone, but not guaranteed to be increasing.\n\n\n\n\n\n","category":"function"},{"location":"#Specific-function-families-1","page":"SpectralKit","title":"Specific function families","text":"","category":"section"},{"location":"#","page":"SpectralKit","title":"SpectralKit","text":"Chebyshev\nChebyshevSemiInf\nChebyshevInf\nChebyshevInterval","category":"page"},{"location":"#SpectralKit.Chebyshev","page":"SpectralKit","title":"SpectralKit.Chebyshev","text":"struct Chebyshev <: SpectralKit.FunctionFamily\n\nChebyhev polynomials of the first kind, defined on [-1,1].\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.ChebyshevSemiInf","page":"SpectralKit","title":"SpectralKit.ChebyshevSemiInf","text":"struct ChebyshevSemiInf{T<:Real} <: SpectralKit.TransformedChebyshev\n\nChebyshev polynomials transformed to the domain [A, Inf) (when L > 0) or (-Inf,A] (when L < 0) using y = A + L  (1 + x)  (1 - x).\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.ChebyshevInf","page":"SpectralKit","title":"SpectralKit.ChebyshevInf","text":"struct ChebyshevInf{T<:Real} <: SpectralKit.TransformedChebyshev\n\nChebyshev polynomials transformed to the domain (-Inf, Inf). (when L < 0) using y = A + L  x  (L^2 + x^2).\n\n\n\n\n\n","category":"type"},{"location":"#SpectralKit.ChebyshevInterval","page":"SpectralKit","title":"SpectralKit.ChebyshevInterval","text":"struct ChebyshevInterval{T<:Real} <: SpectralKit.TransformedChebyshev\n\nChebyshev polynomials transformed to the domain (a, b). using y = x  s + m.\n\nm and s are calculated and checked by the constructor; a < b is enforced.\n\n\n\n\n\n","category":"type"}]
}
