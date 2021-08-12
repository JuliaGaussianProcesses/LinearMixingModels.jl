"""
    ILMM(fs, H)
An Instantaneous Linear Mixing Model (ILMM) -- a distribution over vector-
valued functions. Let `p` be the number of observed outputs, and `m` the number of latent
processes, then
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

const isotopic_inputs = Union{
    KernelFunctions.MOInputIsotopicByFeatures,
    KernelFunctions.MOInputIsotopicByOutputs
}

"""
    get_latent_gp(f::ILMM)

Returns the underlying latent space AbstractGP belonging to `f`.


```jldoctest
julia> f = ILMM(IndependentMOGP([GP(SEKernel())]), rand(2,2));

julia> latent_f = get_latent_gp(f);

julia> latent_f == IndependentMOGP([GP(SEKernel())])
true
```
"""
get_latent_gp(f::ILMM) = f.f

"""
    noise_var(Σ)

Return the diagonal element of Σ.

```jldoctest
julia> Σ = Diagonal(Fill(2, 3));

julia> noise_var(Σ) == 2
true
```
"""
noise_var(Σ::Diagonal{<:Real, <:Fill}) = FillArrays.getindex_value(Σ.diag)


"""
    reshape_y(y, N)

Reshape `y` in to an adjoint Matrix of dimension (length(y)/N, N)`

```jldoctest
julia> y = rand(16);

julia> size(reshape_y(y,8)) == (2, 8)
true

julia> size(reshape_y(y,2)) == (8, 2)
true
```
"""
reshape_y(y::AbstractVector{<:Real}, N::Int) = reshape(y, N, :)'

"""
    unpack(fx)

Collect the relevant underlying fields of the Finite ILMM. This includes
the latent space GP, the mixing matrix, the noise and the observations.

```jldoctest
julia> fs = independent_mogp([GP(Matern32Kernel())]);;

julia> H = rand(2,1);

julia> x = KernelFunctions.MOInputIsotopicByFeatures(ColVecs(rand(2,2), 2));

julia> ilmmx = ILMM(fs, H)(x, 0.1);

julia> (fs, H, 0.1, x.x) == lmm.unpack(ilmmx)
true
```
"""
function unpack(fx::FiniteGP{<:ILMM, <:isotopic_inputs, <:Diagonal{<:Real, <:Fill}})
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
    f_latent, H, σ², x = unpack(fx)

    x_mo_input = KernelFunctions.MOInputIsotopicByFeatures(x, size(H,2))

    latent_rand =  rand(rng, f_latent(x_mo_input))
    return vec(H * reshape(latent_rand, :, length(fx.x.x)))
end

AbstractGPs.rand(fx::FiniteGP{<:ILMM}) = rand(Random.GLOBAL_RNG, fx)

function AbstractGPs.rand(rng::AbstractRNG, fx::FiniteGP{<:ILMM}, N::Int)
    return reduce(hcat, [rand(rng, fx) for _ in 1:N])
end

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

"""
    marginals(fx::FiniteGP{<:ILMM})

Returns the marginal distribution over the ILMM without the IID noise components.
See AbstractGPs.jl API docs.
"""
function AbstractGPs.marginals(fx::FiniteGP{<:ILMM})
    f, H, σ², x = unpack(fx)

    x_mo_input = KernelFunctions.MOInputIsotopicByFeatures(x, size(H,2))

    # Compute the variances.
    M, V = mean_and_var(fx)

    # Package everything into independent Normal distributions.
    return Normal.(M, sqrt.(V))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.mean_and_var(fx::FiniteGP{<:ILMM})
    f, H, σ², x = unpack(fx)

    x_mo_input = KernelFunctions.MOInputIsotopicByFeatures(x, size(H,2))

    latent_mean, latent_var = mean_and_var(f(x_mo_input))

    M = (H * reshape(latent_mean, :, length(x)))'
    V = (abs2.(H) * reshape(latent_var, :, length(x)))' .+ σ²
    return collect(vec(M)), collect(vec(V))
end

# See AbstractGPs.jl API docs.
AbstractGPs.mean(fx::FiniteGP{<:ILMM}) = mean_and_var(fx)[1]

# See AbstractGPs.jl API docs.
AbstractGPs.var(fx::FiniteGP{<:ILMM}) = mean_and_var(fx)[2]

# See AbstractGPs.jl API docs.
function AbstractGPs.logpdf(fx::FiniteGP{<:ILMM}, y::AbstractVector{<:Real})
    f, H, σ², x = unpack(fx)
    p, m = size(H)

    # # Projection step.
    Y = reshape_y(y, length(x))
    T, ΣT = project(H, σ²)
    Ty = ColVecs(T*Y)
    Xproj, Yproj = prepare_isotopic_multi_output_data(x, Ty)
    ΣT = BlockDiagonal([ΣT for _ in 1:length(x)])

    return AbstractGPs.logpdf(f(Xproj, ΣT), Yproj) + regulariser(fx, ColVecs(Y))
end

"""
    regulariser(fx, y)

Computes the regularisation term of the logpdf.
See e.g. appendix A.4 of [1] - Bruinsma et al 2020.
"""
function regulariser(fx, y::ColVecs{<:Real})
    fs, H, σ², x = unpack(fx)
    p, m = size(H)

    # Projection step.
    Y = reshape_y(vec(y.X), length(x))
    T, ΣT = project(H, σ²)

    return -((p - m) * log(2π) + (p * log(σ²) - logdet(ΣT)) +
        sum((Y .- H*T*Y)' * (1/σ²) * (Y .- H*T*Y))) / 2
end

function regulariser(fx, Y::RowVecs{<:Real})
    return regulariser(fx, ColVecs(Y.X'))
end

# See AbstractGPs.jl API docs.
function AbstractGPs.posterior(fx::FiniteGP{<:ILMM}, y::AbstractVector{<:Real})
    f, H, σ², x = unpack(fx)
    p, m = size(H)

    # # Projection step.
    Y = reshape_y(y, length(x))
    T, ΣT = project(H, σ²)
    Ty = ColVecs(T*Y)
    Xproj, Yproj = prepare_isotopic_multi_output_data(x, Ty)
    ΣT = BlockDiagonal([ΣT for _ in 1:length(x)])

    f_posterior = AbstractGPs.posterior(f(Xproj, ΣT), Yproj)
    return ILMM(f_posterior, H)
end
