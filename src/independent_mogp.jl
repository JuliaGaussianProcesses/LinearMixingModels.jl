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
independent_mogp(fs::Vector{<:AbstractGP}) = IndependentMOGP(fs)

"""
    finite_gps(fx)

Returns a list of of the finite GPs for all latent processes, given a finite
IndependentMOGP and *isotopic inputs*.
"""
function finite_gps(fx::FiniteGP{<:IndependentMOGP, <:MOInputIsotopicByOutputs}, σ²::Real)
    return [f(fx.x.x, σ²) for f in fx.f.fs]
end

const IsotropicByOutputsFiniteIndependentMOGP = FiniteGP{
    <:IndependentMOGP, <:MOInputIsotopicByOutputs, <:Diagonal{<:Real, <:Fill},
}

# Optimisations for MOInputIsotopicByOutputs.

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
    return Matrix(BlockDiagonal(Cs))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.cov(
    f::IndependentMOGP, x::MOInputIsotopicByOutputs, y::MOInputIsotopicByOutputs
)
    Cs = map(f -> cov(f, x.x, y.x), f.fs)
    return Matrix(BlockDiagonal(Cs))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.logpdf(ft::IsotropicByOutputsFiniteIndependentMOGP, y::AbstractVector)
    finiteGPs = finite_gps(ft, ft.Σy[1])
    ys = collect(eachcol(reshape(y, (length(ft.x.x), :))))
    return sum(map(logpdf, finiteGPs, ys))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.rand(rng::AbstractRNG, ft::IsotropicByOutputsFiniteIndependentMOGP)
    finiteGPs = finite_gps(ft, ft.Σy[1])
    return reduce(vcat, map(fx -> rand(rng, fx), finiteGPs))
end

# See AbstractGPs.jl API docs.
AbstractGPs.rand(ft::FiniteGP{<:IndependentMOGP}) = rand(Random.GLOBAL_RNG, ft)

# See AbstractGPs.jl API docs.
function AbstractGPs.rand(
    rng::AbstractRNG, ft::IsotropicByOutputsFiniteIndependentMOGP, N::Int,
)
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
    ft::IsotropicByOutputsFiniteIndependentMOGP, y::AbstractVector{<:Real}
)
    finiteGPs = finite_gps(ft, ft.Σy[1])
    ys = collect(eachcol(reshape(y, (length(ft.x.x), :))))
    ind_posts = [AbstractGPs.posterior(fx, y_i) for (fx, y_i) in zip(finiteGPs, ys)]
    return independent_mogp(ind_posts)
end


# AbstractGPs APIs implementations for MOInputIsotopicByFeatures.

const IsotropicByFeaturesFiniteIndependentMOGP = FiniteGP{
    <:IndependentMOGP, <:MOInputIsotopicByFeatures, <:Diagonal{<:Real, <:Fill},
}

function reorder_features_to_outputs_indices(x::MOInputIsotopicByOutputs)
    return sortperm(vec(reshape(1:length(x), x.out_dim, length(x.x))'))
end

function reorder_features_to_outputs_indices(x::MOInputIsotopicByFeatures)
    return vec(reshape(1:length(x), x.out_dim, length(x.x))')
end

reorder_by_outputs(x::MOInputIsotopicByFeatures) = MOInputIsotopicByOutputs(x.x, x.out_dim)

# We know that the observation noise covariance is constant-diagonal, so no need to reorder.
function reorder_by_outputs(fx::IsotropicByFeaturesFiniteIndependentMOGP)
    return FiniteGP(fx.f, reorder_by_outputs(fx.x), fx.Σy)
end

ChainRulesCore.@non_differentiable reorder_features_to_outputs_indices(::Any)
ChainRulesCore.@non_differentiable reorder_by_outputs(::Any)

function finite_gps(fx::FiniteGP{<:IndependentMOGP, <:MOInputIsotopicByFeatures}, σ²::Real)
    return [f(fx.x.x, σ²) for f in fx.f.fs]
end

function AbstractGPs.mean(f::IndependentMOGP, x::MOInputIsotopicByFeatures)
    x_by_outputs = reorder_by_outputs(x)
    mean_by_outputs = mean(f, x_by_outputs)
    return mean_by_outputs[reorder_features_to_outputs_indices(x_by_outputs)]
end

function AbstractGPs.var(f::IndependentMOGP, x::MOInputIsotopicByFeatures)
    x_by_outputs = reorder_by_outputs(x)
    var_by_outputs = var(f, x_by_outputs)
    return var_by_outputs[reorder_features_to_outputs_indices(x_by_outputs)]
end

function AbstractGPs.cov(f::IndependentMOGP, x::MOInputIsotopicByFeatures)
    x_by_outputs = reorder_by_outputs(x)
    C_by_outputs = cov(f, x_by_outputs)
    idx = reorder_features_to_outputs_indices(x_by_outputs)
    return C_by_outputs[idx, idx]
end

function AbstractGPs.rand(rng::AbstractRNG, ft::IsotropicByFeaturesFiniteIndependentMOGP)
    finiteGPs = finite_gps(ft, ft.Σy[1])
    return vec(reduce(hcat, map(fx -> rand(rng, fx), finiteGPs))')
end

function AbstractGPs.logpdf(
    ft::IsotropicByFeaturesFiniteIndependentMOGP, y::AbstractVector{<:Real},
)
    return logpdf(reorder_by_outputs(ft), y[reorder_features_to_outputs_indices(ft.x)])
end
