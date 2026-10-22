using Documenter, VSAs

# Code run before every jldoctest block, so examples don't need their own `using`.
DocMeta.setdocmeta!(VSAs, :DocTestSetup, :(using VSAs, Distributions, Random); recursive=true)

makedocs(;
    sitename="VSAs.jl",
    modules=[VSAs],
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true", edit_link=nothing),
    pages=["Home" => "index.md"],
    doctest=true,
    # No git remote yet; remove this once the repo is on GitHub so source links work.
    remotes=nothing,
)
