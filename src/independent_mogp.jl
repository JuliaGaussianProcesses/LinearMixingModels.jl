"""
    IndependentMOGP
"""
struct IndependentMOGP{Tg<:Vector{<:AbstractGPs.AbstractGP}} <: AbstractGPs.AbstractGP
    fs::Tg
end

"""
    independent_mogp

Returns an IndependentMOGP given a list of AbstractGPs.
"""
function independent_mogp(fs::Vector{<:AbstractGPs.AbstractGP})
    return IndependentMOGP(fs)
end

"""
A constant to represent all isotopic input types.
"""
const isotopic_inputs = Union{
    KernelFunctions.MOInputIsotopicByFeatures, KernelFunctions.MOInputIsotopicByOutputs
}

"""
    finite_gps()

Returns a list of of the finite GPs for all latent processes.
"""
function finite_gps(ft::FiniteGP{<:IndependentMOGP,<:isotopic_inputs})
    return [f(ft.x.x, ft.Σy[1:length(ft.x.x),1:length(ft.x.x)]) for f in ft.f.fs]
end

"""
    finite_gps()

Returns a list of of the finite GPs for all latent processes for some input x.
"""
function finite_gps(ft::FiniteGP{<:IndependentMOGP,<:isotopic_inputs}, x::AbstractVector)
    return [f(x, ft.Σy[1:length(ft.x.x),1:length(ft.x.x)]) for f in ft.f.fs]
end

# Implement AbstractGPs API

function AbstractGPs.marginals(ft::FiniteGP{<:IndependentMOGP})
    finiteGPs = finite_gps(ft)
    return reduce(hcat, map(marginals, finiteGPs))
end

function AbstractGPs.mean_and_var(ft::FiniteGP{<:IndependentMOGP})
    ms = marginals(ft)
    return map(mean, ms), map(var, ms)
end

AbstractGPs.mean(ft::FiniteGP{<:IndependentMOGP}) = mean_and_var(ft)[1]

AbstractGPs.var(ft::FiniteGP{<:IndependentMOGP}) = mean_and_var(ft)[2]

AbstractGPs.cov(ft::FiniteGP{<:IndependentMOGP}) = cov.(FiniteGP.(ft.f.fs, ft.x, ft.Σy))

function AbstractGPs.logpdf(ft::FiniteGP{<:IndependentMOGP}, y::AbstractVector{<:Union{Missing, Real}})
    finiteGPs = finite_gps(ft)
    return [AbstractGPs.logpdf(f, y) for f in finiteGPs]
end

##########################

function AbstractGPs.rand(rng::AbstractRNG, ft::FiniteGP{<:IndependentMOGP})
    return destructure(ft.x, rand(rng, ft))
end

AbstractGPs.rand(ft::FiniteGP{<:IndependentMOGP}) = rand(Random.GLOBAL_RNG, ft)

function AbstractGPs.rand(rng::AbstractRNG, ft::FiniteGP{<:IndependentMOGP}, N::Int)
    finiteGPs = finite_gps(ft)
    return [rand(rng, f) for f in finiteGPs]
end

AbstractGPs.rand(ft::FiniteGP{<:IndependentMOGP}, N::Int) = rand(Random.GLOBAL_RNG, ft, N)
