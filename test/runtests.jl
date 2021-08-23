using AbstractGPs
using ChainRules
using ChainRulesCore
using ChainRulesTestUtils
using Distributions
using Documenter
using FillArrays
using KernelFunctions
using LinearAlgebra
using LinearMixingModels
using Random
using Test
using Zygote

using AbstractGPs: AbstractGP, FiniteGP
using AbstractGPs.TestUtils:
    test_finitegp_primary_public_interface,
    test_finitegp_primary_and_secondary_public_interface,
    test_internal_abstractgps_interface
using KernelFunctions: MOInputIsotopicByOutputs
using LinearMixingModels: unpack, noise_var, get_latent_gp, reshape_y

function _is_approx(x::AbstractVector{<:Normal}, y::AbstractVector{<:Normal})
    return (map(mean, x) ≈ map(mean, y)) && (map(std, x) ≈ map(std, y))
end

const ad_config = Zygote.ZygoteRuleConfig()

function ChainRulesTestUtils.rand_tangent(
    rng::AbstractRNG, x::T,
) where {T<:MOInputIsotopicByOutputs}
    return Tangent{T}(x=rand_tangent(rng, x.x))
end

include("test_utils.jl")

@testset "LinearMixingModels.jl" begin
    include("independent_mogp.jl")
    # include("orthogonal_matrix.jl")
    # include("ilmm.jl")
    # include("oilmm.jl")

    @testset "doctests" begin
        DocMeta.setdocmeta!(
            LinearMixingModels,
            :DocTestSetup,
            quote
                using AbstractGPs
                using KernelFunctions
                using LinearMixingModels
                using Random
                using LinearAlgebra
                using FillArrays
            end;
            recursive=true,
        )
        doctest(
            LinearMixingModels;
            doctestfilters=[
                r"{([a-zA-Z0-9]+,\s?)+[a-zA-Z0-9]+}",
                r"(Array{[a-zA-Z0-9]+,\s?1}|\s?Vector{[a-zA-Z0-9]+})",
                r"(Array{[a-zA-Z0-9]+,\s?2}|\s?Matrix{[a-zA-Z0-9]+})",
            ],
        )
    end
    @info "Ran doctests."
end
