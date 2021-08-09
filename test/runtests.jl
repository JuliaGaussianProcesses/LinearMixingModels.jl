using AbstractGPs
using Distributions
using FiniteDifferences
using LinearAlgebra
# using LinearMixingModels
using Random
using Stheno
using Test
using Zygote

using AbstractGPs: AbstractGP, FiniteGP
# using LinearMixingModxels: marginals, rand
using Stheno: GaussianProcessProbabilisticProgramme

# Helper functionality, doesn't actually run any tests.
# include("test_util.jl")

@testset "LinearMixingModels.jl" begin
    include("oilmm.jl")
    include("ilmm.jl")
    include("independent_mogp.jl")
    include("orthogonal_matrix.jl")
end
