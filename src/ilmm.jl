"""
    ILMM(fs, H)
An Instantaneous Linear Mixing Model (ILMM) -- a distribution over vector-
valued functions. Let `p` be the number of observed outputs, and `m` the number of latent
processes, then `H`, also known as the mixing matrix, is a `p x m` matrix whose column
space spans the output space. The latent processes are represented a Gaussian process `f`.

# Arguments:
- fs: a length-`m` vector of Gaussian process objects from AbstractGPs.jl.
- H: a `p x m` matrix representing a fixed basis of our p-dim target: h_1,...,h_m

"""
struct ILMM{
    Tf<:AbstractGP,
    TH<:AbstractMatrix{<:Real}
} <: AbstractGP
    f::Tf
    H::TH
end

"""
    get_latent_gp(f::ILMM)

Returns the underlying latent space AbstractGP belonging to `f`.


```jldoctest
julia> f = ILMM(IndependentMOGP([GP(SEKernel())]), rand(2,2));

julia> latent_f = get_latent_gp(f);

julia> latent_f isa IndependentMOGP
true

julia> latent_f.fs == [GP(SEKernel())]
true
```
"""
get_latent_gp(f::ILMM) = f.f

noise_var(Σ::Diagonal{<:Real, <:Fill}) = FillArrays.getindex_value(Σ.diag)

reshape_y(y::AbstractVector{<:Real}, N::Int) = reshape(y, N, :)'

function unpack(fx::FiniteGP{<:ILMM, <:MOInputIsotopicByOutputs, <:Diagonal{<:Real, <:Fill}})
    f = fx.f.f
    H = fx.f.H
    σ² = noise_var(fx.Σy)
    x = fx.x.x

    # Check that the number of outputs requested agrees with the model.
    fx.x.out_dim == size(H, 1) || throw(error("out dim of x != out dim of f."))
    return f, H, σ², x
end

"""
    project(H, σ²)

Computes the projection `T` and `ΣT` given the mixing matrix and noise.
"""
function project(H::AbstractMatrix{Z}, σ²::Z) where {Z<:Real}
    Σ = σ²*I

    # Compute the projected noise, which is a matrix of size `size(Yproj)`.
    ΣT = inv(H' * inv(Σ) * H)

    # Compute transformation for the projection of the data.
    T = ΣT * H' * inv(Σ)

    return T, ΣT
end

"""
    rand(rng::AbstractRNG, fx::FiniteGP{<:ILMM})

Sample from the ILMM, including the observation noise.
Follows generative structure of model 2 from [1].
Follows the AbstractGPs.jl API.
[1] - Bruinsma et al 2020.
"""
function AbstractGPs.rand(rng::AbstractRNG, fx::FiniteGP{<:ILMM})
    f, H, σ², x = unpack(fx)
    p, m = size(H)

    x_mo_input = MOInputIsotopicByOutputs(x, m)

    latent_rand =  rand(rng, f(x_mo_input))
    return vec(H * reshape(latent_rand, :, length(x)))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.rand(rng::AbstractRNG, fx::FiniteGP{<:ILMM}, N::Int)
    return reduce(hcat, [rand(rng, fx) for _ in 1:N])
end

# See AbstractGPs.jl API docs.
function Distributions._rand!(
    rng::AbstractRNG,
    fx::FiniteGP{<:ILMM},
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
function AbstractGPs.mean_and_var(fx::FiniteGP{<:ILMM})
    f, H, σ², x = unpack(fx)
    p, m = size(H)
    n = length(x)

    x_mo_input = MOInputIsotopicByOutputs(x, m)

    latent_mean, latent_cov = mean_and_cov(f(x_mo_input))

    H_full = kron(H, Matrix(I,n,n))

    M = H_full * latent_mean
    V = diag(H_full * latent_cov * H_full') .+ σ²

    return collect(vec(M)), V
end

# See AbstractGPs.jl API docs.
AbstractGPs.mean(fx::FiniteGP{<:ILMM}) = mean_and_var(fx)[1]

# See AbstractGPs.jl API docs.
AbstractGPs.var(fx::FiniteGP{<:ILMM}) = mean_and_var(fx)[2]

# See AbstractGPs.jl API docs.
function AbstractGPs.logpdf(fx::FiniteGP{<:ILMM}, y::AbstractVector{<:Real})
    f, H, σ², x = unpack(fx)
    p, m = size(H)
    n = length(x)

    # # Projection step.
    Y = reshape_y(y, length(x))
    T, ΣT = project(H, σ²)
    Ty = RowVecs((T*Y)')
    Xproj, Yproj = prepare_isotopic_multi_output_data(x, Ty)
    ΣT = kron(ΣT, Matrix(I,n,n))

    return AbstractGPs.logpdf(f(Xproj, ΣT), Yproj) + regulariser(fx, Y)
end

"""
    regulariser(fx, y)

Computes the regularisation term of the logpdf.
See e.g. appendix A.4 of [1] - Bruinsma et al 2020.
"""
function regulariser(fx, Y::AbstractMatrix{<:Real})
    fs, H, σ², x = unpack(fx)
    p, m = size(H)
    n = size(Y, 2)

    # Projection step.
    T, ΣT = project(H, σ²)

    return -(n * ((p - m) * log(2π) + (p * log(σ²) - logdet(ΣT))) +
        sum((1/σ²) * abs2.(Y .- H*T*Y))) / 2
end

# See AbstractGPs.jl API docs.
function AbstractGPs.posterior(fx::FiniteGP{<:ILMM}, y::AbstractVector{<:Real})
    f, H, σ², x = unpack(fx)
    p, m = size(H)
    n = length(x)

    # # Projection step.
    Y = reshape_y(y, length(x))
    T, ΣT = project(H, σ²)
    Ty = RowVecs((T*Y)')
    Xproj, Yproj = prepare_isotopic_multi_output_data(x, Ty)
    ΣT = kron(ΣT, Matrix(I,n,n))

    f_posterior = AbstractGPs.posterior(f(Xproj, ΣT), Yproj)
    return ILMM(f_posterior, H)
end
