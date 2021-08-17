using AbstractGPs
using Distributions
# using FiniteDifferences
using Documenter
using KernelFunctions
using LinearAlgebra
using LinearMixingModels
using Random
using Stheno
using Test
using Zygote

using AbstractGPs: AbstractGP, FiniteGP
using AbstractGPs.TestUtils: test_finitegp_primary_public_interface
using KernelFunctions: MOInputIsotopicByFeatures, MOInputIsotopicByOutputs
# using LinearMixingModxels: marginals, rand
# using Stheno: GaussianProcessProbabilisticProgramme

@testset "LinearMixingModels.jl" begin
    @testset "oilmm" begin
        include("oilmm.jl")
    end
    @info "Ran oilmm tests."

    @testset "ilmm" begin
        include("ilmm.jl")
    end
    @info "Ran ilmm tests."

    @testset "independent_mogp" begin
        include("independent_mogp.jl")
    end
    @info "Ran independent_mogp tests."

    @testset "orthogonal_matrix" begin
        include("orthogonal_matrix.jl")
    end
    @info "Ran orthogonal_matrix tests."

    # @testset "doctests" begin
    #     DocMeta.setdocmeta!(
    #         LinearMixingModels,
    #         :DocTestSetup,
    #         :(using AbstractGPs, Random, LinearAlgebra);
    #         recursive=true,
    #     )
    #     doctest(
    #         LinearMixingModels;
    #         doctestfilters=[
    #             r"{([a-zA-Z0-9]+,\s?)+[a-zA-Z0-9]+}",
    #             r"(Array{[a-zA-Z0-9]+,\s?1}|\s?Vector{[a-zA-Z0-9]+})",
    #             r"(Array{[a-zA-Z0-9]+,\s?2}|\s?Matrix{[a-zA-Z0-9]+})",
    #         ],
    #     )
    # end
end
