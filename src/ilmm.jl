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
    MOInputIsotopicByFeatures,
    MOInputIsotopicByOutputs
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

julia> size(LinearMixingModels.reshape_y(y,8)) == (2, 8)
true

julia> size(LinearMixingModels.reshape_y(y,2)) == (8, 2)
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

julia> x = MOInputIsotopicByOutputs(ColVecs(rand(2,2), 2));

julia> ilmmx = ILMM(fs, H)(x, 0.1);

julia> (fs, H, 0.1, x.x) == LinearMixingModels.unpack(ilmmx)
true
```
"""
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

# """
#     marginals(fx::FiniteGP{<:ILMM})

# Returns the marginal distribution over the ILMM without the IID noise components.
# See AbstractGPs.jl API docs.
# """
# function AbstractGPs.marginals(fx::FiniteGP{<:ILMM})
#     f, H, σ², x = unpack(fx)
#     p, m = size(H)

#     x_mo_input = MOInputIsotopicByOutputs(x, m)

#     # Compute the variances.
#     M, V = mean_and_var(fx)

#     # Package everything into independent Normal distributions.
#     return Normal.(M, sqrt.(V))
# end

# See AbstractGPs.jl API docs.
function AbstractGPs.mean_and_var(fx::FiniteGP{<:ILMM})
    f, H, σ², x = unpack(fx)
    p, m = size(H)
    n = length(x)

    x_mo_input = MOInputIsotopicByOutputs(x, m)

    # wrong, needs fixing
    latent_mean, latent_cov = mean_and_cov(f(x_mo_input))

    H_full = kron(H, Matrix(I,n,n))
    # H_block =  BlockDiagonal([H for _ in 1:n])
    # H_block′ =  BlockDiagonal([H' for _ in 1:n])
    # @show size(H_block)
    # @show size(H_block′)
    # @show size(latent_cov)

    # M = (H * reshape(latent_mean, :, length(x)))'
    # Compute the variances.
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

    # @show (sum((1/σ²) * abs2.(Y .- H*T*Y))) / 2
    # @show -(n * ((p - m) * log(2π) + (p * log(σ²) - logdet(ΣT))))/2

    return -(n * ((p - m) * log(2π) + (p * log(σ²) - logdet(ΣT))) +
        sum((1/σ²) * abs2.(Y .- H*T*Y))) / 2
end

# function regulariser(fx, Y::RowVecs{<:Real})
#     return regulariser(fx, ColVecs(Y.X'))
# end

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
