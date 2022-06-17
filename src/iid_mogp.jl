"""
    IIDMOGP(f::AbstractGP, num_outputs::Int)

A multi-output GP with `num_outputs` where the GP for each output has the same distribution,
given by `f` -- that is, each output is independently and identically distributed (IID).
"""
struct IIDMOGP{Tf<:AbstractGP} <: AbstractGP
    f::Tf
    num_outputs::Int
end


# Optimisations for MOInputIsotopicByOutputs.

function check_inputs(f::IIDMOGP, x::MOInputIsotopicByOutputs)
    if f.num_outputs != x.out_dim
        throw(error("f has $(f.num_outputs) outputs, but x only $(x.out_dim)."))
    end
end

@non_differentiable check_inputs(::Any, ::Any)

# See AbstractGPs.jl API docs.
function AbstractGPs.mean(f::IIDMOGP, x::MOInputIsotopicByOutputs)
    check_inputs(f, x)
    return repeat(mean(f.f, x.x), x.out_dim)
end

# See AbstractGPs.jl API docs.
function AbstractGPs.var(f::IIDMOGP, x::MOInputIsotopicByOutputs)
    check_inputs(f, x)
    return repeat(var(f.f, x.x), x.out_dim)
end

# See AbstractGPs.jl API docs.
function AbstractGPs.cov(f::IIDMOGP, x::MOInputIsotopicByOutputs)
    check_inputs(f, x)
    return Matrix(BlockDiagonal(fill(cov(f.f, x.x), x.out_dim)))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.cov(
    f::IIDMOGP, x::MOInputIsotopicByOutputs, y::MOInputIsotopicByOutputs
)
    check_inputs(f, x)
    check_inputs(f, y)
    return Matrix(BlockDiagonal(fill(cov(f, x.x, y.x), x.out_dim)))
end

const IsotropicByOutputsFiniteIIDMOGP = FiniteGP{
    <:IIDMOGP,<:MOInputIsotopicByOutputs,<:Diagonal{<:Real,<:Fill}
}

get_single_gp(fx::IsotropicByOutputsFiniteIIDMOGP) = fx.f(fx.x, fx.Σ.value)

# See AbstractGPs.jl API docs.
function AbstractGPs.logpdf(fx::IsotropicByOutputsFiniteIIDMOGP, y::AbstractVector{<:Real})
    check_inputs(fx.f, fx.x)
    return logpdf(get_single_gp(fx), reshape(y, :, fx.f.num_outputs))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.rand(rng::AbstractRNG, fx::IsotropicByOutputsFiniteIIDMOGP)
    return vec(rand(rng, get_single_gp(fx), fx.f.num_outputs))
end

# See AbstractGPs.jl API docs.
AbstractGPs.rand(fx::FiniteGP{<:IIDMOGP}) = rand(Random.GLOBAL_RNG, fx)

# See AbstractGPs.jl API docs.
function AbstractGPs.rand(
    rng::AbstractRNG, fx::IsotropicByOutputsFiniteIIDMOGP, N::Int
)
    return reduce(hcat, [rand(rng, fx) for _ in 1:N])
end

# See AbstractGPs.jl API docs.
AbstractGPs.rand(fx::FiniteGP{<:IIDMOGP}, N::Int) = rand(Random.GLOBAL_RNG, fx, N)

# See AbstractGPs.jl API docs.
function Distributions._rand!(
    rng::AbstractRNG,
    fx::FiniteGP{<:IIDMOGP},
    y::AbstractVecOrMat{<:Real}
)
    N = size(y, 2)
    if N == 1
        y .= AbstractGPs.rand(rng, fx)
    else
        y .= AbstractGPs.rand(rng, fx, N)
    end
end

# See AbstractGPs.jl API docs.
function AbstractGPs.posterior(
    fx::IsotropicByOutputsFiniteIIDMOGP, y::AbstractVector{<:Real}
)
    finiteGPs = finite_gps(ft, ft.Σy[1])
    ys = collect(eachcol(reshape(y, (length(ft.x.x), :))))
    ind_posts = [AbstractGPs.posterior(fx, y_i) for (fx, y_i) in zip(finiteGPs, ys)]
    return independent_mogp(ind_posts)
end
