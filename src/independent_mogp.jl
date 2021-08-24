"""
    IndependentMOGP(fs)

A multi-output GP with independent outputs where output `i` is modelled by the
single-output GP fs[i].

# Arguments:
- fs: a vector of `p` single-output GPs where `p` is the dimension of the output.
"""
struct IndependentMOGP{Tfs<:Vector{<:AbstractGP}} <: AbstractGP
    fs::Tfs
end

"""
    independent_mogp(fs)

Returns an IndependentMOGP given a list of single output GPs `fs`.

```jldoctest
julia> ind_mogp1 = independent_mogp([GP(KernelFunctions.SEKernel())]);

julia> ind_mogp2 = IndependentMOGP([GP(KernelFunctions.SEKernel())]);

julia> typeof(ind_mogp1) == typeof(ind_mogp2)
true

julia> ind_mogp1.fs == ind_mogp2.fs
true
```
"""
function independent_mogp(fs::Vector{<:AbstractGP})
    return IndependentMOGP(fs)
end

"""
    finite_gps(fx)

Returns a list of of the finite GPs for all latent processes, given a finite
IndependentMOGP and *isotopic inputs*.
"""
function finite_gps(fx::FiniteGP{<:IndependentMOGP, <:MOInputIsotopicByOutputs}, σ²::Real)
    return [f(fx.x.x, σ²) for f in fx.f.fs]
end

const IsotropicFiniteIndependentMOGP = FiniteGP{
    <:IndependentMOGP, <:MOInputIsotopicByOutputs, <:Diagonal{<:Real, <:Fill},
}

# Implement AbstractGPs API

# See AbstractGPs.jl API docs.
function AbstractGPs.mean(f::IndependentMOGP, x::MOInputIsotopicByOutputs)
    return reduce(vcat, map(f -> mean(f, x.x), f.fs))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.var(f::IndependentMOGP, x::MOInputIsotopicByOutputs)
    return reduce(vcat, map(f -> var(f, x.x), f.fs))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.cov(f::IndependentMOGP, x::MOInputIsotopicByOutputs)
    Cs = map(f -> cov(f, x.x), f.fs)
    return collect(BlockDiagonal(Cs))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.cov(
    f::IndependentMOGP,
    x::MOInputIsotopicByOutputs,
    y::MOInputIsotopicByOutputs,
)
    Cs = map(f -> cov(f, x.x, y.x), f.fs)
    return collect(BlockDiagonal(Cs))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.logpdf(ft::IsotropicFiniteIndependentMOGP, y::AbstractVector)
    finiteGPs = finite_gps(ft, ft.Σy[1])
    ys = collect(eachcol(reshape(y, (length(ft.x.x), :))))
    return sum(map(logpdf, finiteGPs, ys))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.rand(rng::AbstractRNG, ft::IsotropicFiniteIndependentMOGP)
    finiteGPs = finite_gps(ft, ft.Σy[1])
    return reduce(vcat, map(fx -> rand(rng, fx), finiteGPs))
end

# See AbstractGPs.jl API docs.
AbstractGPs.rand(ft::FiniteGP{<:IndependentMOGP}) = rand(Random.GLOBAL_RNG, ft)

# See AbstractGPs.jl API docs.
function AbstractGPs.rand(rng::AbstractRNG, ft::IsotropicFiniteIndependentMOGP, N::Int)
    return reduce(hcat, [rand(rng, ft) for _ in 1:N])
end

# See AbstractGPs.jl API docs.
AbstractGPs.rand(ft::FiniteGP{<:IndependentMOGP}, N::Int) = rand(Random.GLOBAL_RNG, ft, N)

# See AbstractGPs.jl API docs.
function Distributions._rand!(
    rng::AbstractRNG,
    fx::FiniteGP{<:IndependentMOGP},
    y::AbstractVecOrMat{<:Real}
)
    N = size(y, 2)
    if N == 1
        y .= AbstractGPs.rand(rng, fx)
    else
        y .= AbstractGPs.rand(rng, fx, N)
    end
end

"""
Posterior implementation for isotopic inputs, given diagonal Σy (OILMM).
See AbstractGPs.jl API docs.
"""
function AbstractGPs.posterior(
    ft::IsotropicFiniteIndependentMOGP, y::AbstractVector{<:Real}
)
    finiteGPs = finite_gps(ft, ft.Σy[1])
    ys = collect(eachcol(reshape(y, (length(ft.x.x), :))))
    ind_posts = [AbstractGPs.posterior(fx, y_i) for (fx, y_i) in zip(finiteGPs, ys)]
    return independent_mogp(ind_posts)
end
