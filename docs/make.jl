using StructuredProximalOperators
using Documenter

makedocs(;
    modules = [StructuredProximalOperators],
    authors = "Gilles Bareilles <gilles.bareilles@protonmail.com> and contributors",
    repo = "https://github.com/GillesBareilles/StructuredProximalOperators.jl/blob/{commit}{path}#L{line}",
    sitename = "StructuredProximalOperators.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://GillesBareilles.github.io/StructuredProximalOperators.jl",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/GillesBareilles/StructuredProximalOperators.jl")
