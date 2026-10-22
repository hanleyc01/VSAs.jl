using Test, Aqua, JET, Documenter, VSAs

DocMeta.setdocmeta!(VSAs, :DocTestSetup, :(using VSAs, Distributions, Random); recursive=true)

@testset "Doctests" begin
    doctest(VSAs; manual=false)
end

@testset "Aqua" begin
    Aqua.test_all(VSAs)
end
@testset "JET" begin
    JET.test_package(VSAs; target_modules=(VSAs,))
end
