using Documenter, SpectralKit
DocMeta.setdocmeta!(SpectralKit, :DocTestSetup, :(using SpectralKit);
                    recursive=true)

makedocs(
    modules = [SpectralKit],
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
    authors = "Tamas K. Papp",
    sitename = "SpectralKit.jl",
    pages = Any["index.md"],
    clean = true,
    checkdocs = :exports,
    warnonly = Documenter.except(:cross_references),
)

deploydocs(
    repo = "github.com/tpapp/SpectralKit.jl.git",
    push_preview = true,
)
