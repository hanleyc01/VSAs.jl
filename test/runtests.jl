using Test, Aqua, JET, Documenter, Random, VSAs

DocMeta.setdocmeta!(VSAs, :DocTestSetup, :(using VSAs, Distributions, Random); recursive=true)

@testset "Doctests" begin
    doctest(VSAs; manual=false)
end

@testset "HRR element types" begin
    rng = Xoshiro(42)
    d = 256

    @testset "plain HRR defaults to Float64" begin
        @test normal(rng, HRR, d) isa HRR{Float64}
        @test unitary(rng, HRR, d) isa HRR{Float64}
        @test normal(HRR, d) isa HRR{Float64}
        @test unitary(HRR, d) isa HRR{Float64}
    end

    @testset "$T stays $T" for T in (Float32, Float64)
        x, y = normal(rng, HRR{T}, d), normal(rng, HRR{T}, d)
        u = unitary(rng, HRR{T}, d)

        @test x isa HRR{T}
        @test u isa HRR{T}
        @test x ⊗ y isa HRR{T}
        @test x ⨸ y isa HRR{T}
        @test x ⊕ y isa HRR{T}
        @test x ⊖ y isa HRR{T}
        @test permute(x) isa HRR{T}
        @test unpermute(x) isa HRR{T}
        @test similarity(x, y) isa T

        # unitary vectors make unbinding exact
        @test similarity((u ⊗ y) ⨸ u, y) > T(0.99)
    end

    @testset "unsupported element types" begin
        @test_throws MethodError normal(rng, HRR{Int}, d)
        @test_throws MethodError unitary(rng, HRR{Float16}, d)
    end
end

@testset "Aqua" begin
    Aqua.test_all(VSAs)
end
@testset "JET" begin
    JET.test_package(VSAs; target_modules=(VSAs,))
end
