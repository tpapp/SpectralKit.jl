using Documenter, SpectralKit, Miter, Colors
DocMeta.setdocmeta!(SpectralKit, :DocTestSetup,
                    quote
                        using SpectralKit
                        using SpectralKit: 𝑑, ∂ # derivatives API currently not exposed
                    end,
                    recursive=true)

makedocs(
    modules = [SpectralKit],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Tamas K. Papp",
    sitename = "SpectralKit.jl",
    pages = Any["overview.md",
                "smolyak.md",
                "approximation.md",
                "api.md"],
    clean = true,
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/tpapp/SpectralKit.jl.git",
    push_preview = true,
)
