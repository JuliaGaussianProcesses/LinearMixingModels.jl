var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = LinearMixingModels","category":"page"},{"location":"#LinearMixingModels","page":"Home","title":"LinearMixingModels","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for LinearMixingModels.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [LinearMixingModels]","category":"page"},{"location":"#LinearMixingModels.ILMM","page":"Home","title":"LinearMixingModels.ILMM","text":"ILMM(fs, H)\n\nAn Instantaneous Linear Mixing Model (ILMM) – a distribution over vector- valued functions. Let p be the number of observed outputs, and m the number of latent processes, then H, also known as the mixing matrix, is a p x m matrix whose column space spans the output space. The latent processes are represented a Gaussian process f.\n\nArguments:\n\nfs: a length-m vector of Gaussian process objects from AbstractGPs.jl.\nH: a p x m matrix representing a fixed basis of our p-dim target: h1,...,hm\n\n\n\n\n\n","category":"type"},{"location":"#LinearMixingModels.IndependentMOGP","page":"Home","title":"LinearMixingModels.IndependentMOGP","text":"IndependentMOGP(fs)\n\nA multi-output GP with independent outputs where output i is modelled by the single-output GP fs[i].\n\nArguments:\n\nfs: a vector of p single-output GPs where p is the dimension of the output.\n\n\n\n\n\n","category":"type"},{"location":"#LinearMixingModels.OILMM","page":"Home","title":"LinearMixingModels.OILMM","text":"OILMM(fs, H)\n\nAn Orthogonal Instantaneous Linear Mixing Model (OILMM) – a distribution over vector- valued functions. Let p be the number of observed outputs, and m the number of latent processes, then H, also known as the mixing matrix, is a p x m orthogonal matrix whose column space spans the output space. The latent processes are represented an Independent_MOGP as the latent processes remain decoupled.\n\nArguments:\n\nf: a length-m vector of Gaussian process objects as an IndependentMOGP.\nH: a p x m orthogonal matrix representing a fixed basis of our p-dim target: h1,...,hm\n\n\n\n\n\n","category":"type"},{"location":"#LinearMixingModels.Orthogonal","page":"Home","title":"LinearMixingModels.Orthogonal","text":"Orthogonal(U, S; validate_fields)\n\nAn AbstractMatrix H that takes the form H = U * sqrt(S) with U a matrix with orthonormal columns and S a diagonal matrix with positive entries.\n\nArguments:\n\nU: a p x m matrix with mutually orthonormal columns.\nS: an m x m Diagonal matrix with positive entries.\n\n\n\n\n\n","category":"type"},{"location":"#AbstractGPs.posterior-Tuple{AbstractGPs.FiniteGP{var\"#s4\", var\"#s3\", var\"#s2\"} where {var\"#s4\"<:IndependentMOGP, var\"#s3\"<:KernelFunctions.MOInputIsotopicByOutputs, var\"#s2\"<:(LinearAlgebra.Diagonal{var\"#s1\", var\"#s6\"} where {var\"#s1\"<:Real, var\"#s6\"<:FillArrays.Fill})}, AbstractVector{var\"#s2\"} where var\"#s2\"<:Real}","page":"Home","title":"AbstractGPs.posterior","text":"Posterior implementation for isotopic inputs, given diagonal Σy (OILMM). See AbstractGPs.jl API docs.\n\n\n\n\n\n","category":"method"},{"location":"#Base.rand-Tuple{Random.AbstractRNG, AbstractGPs.FiniteGP{var\"#s2\", Tx, TΣ} where {var\"#s2\"<:ILMM, Tx<:(AbstractVector{T} where T), TΣ}}","page":"Home","title":"Base.rand","text":"rand(rng::AbstractRNG, fx::FiniteGP{<:ILMM})\n\nSample from the ILMM, including the observation noise. Follows generative structure of model 2 from [1]. Follows the AbstractGPs.jl API. [1] - Bruinsma et al 2020.\n\n\n\n\n\n","category":"method"},{"location":"#Base.rand-Tuple{Random.AbstractRNG, AbstractGPs.FiniteGP{var\"#s4\", Tx, TΣ} where {var\"#s4\"<:(OILMM{var\"#s14\", var\"#s6\"} where {var\"#s14\"<:IndependentMOGP, var\"#s6\"<:Orthogonal}), Tx<:(AbstractVector{T} where T), TΣ}}","page":"Home","title":"Base.rand","text":"rand(rng::AbstractRNG, fx::FiniteGP{<:OILMM})\n\nSample from the OILMM, including the observation noise. Follows generative structure of model 2 from [1]. Follows the AbstractGPs.jl API. See also rand_latent. [1] - Bruinsma et al 2020.\n\n\n\n\n\n","category":"method"},{"location":"#LinearMixingModels.finite_gps-Tuple{AbstractGPs.FiniteGP{var\"#s2\", var\"#s1\", TΣ} where {var\"#s2\"<:IndependentMOGP, var\"#s1\"<:KernelFunctions.MOInputIsotopicByOutputs, TΣ}, Real}","page":"Home","title":"LinearMixingModels.finite_gps","text":"finite_gps(fx)\n\nReturns a list of of the finite GPs for all latent processes, given a finite IndependentMOGP and isotopic inputs.\n\n\n\n\n\n","category":"method"},{"location":"#LinearMixingModels.get_latent_gp-Tuple{ILMM}","page":"Home","title":"LinearMixingModels.get_latent_gp","text":"get_latent_gp(f::ILMM)\n\nReturns the underlying latent space AbstractGP belonging to f.\n\njulia> f = ILMM(independent_mogp([GP(SEKernel())]), rand(2,2));\n\njulia> latent_f = get_latent_gp(f);\n\njulia> latent_f isa IndependentMOGP\ntrue\n\njulia> latent_f.fs == [GP(SEKernel())]\ntrue\n\n\n\n\n\n","category":"method"},{"location":"#LinearMixingModels.independent_mogp-Tuple{Vector{var\"#s3\"} where var\"#s3\"<:AbstractGPs.AbstractGP}","page":"Home","title":"LinearMixingModels.independent_mogp","text":"independent_mogp(fs)\n\nReturns an IndependentMOGP given a list of single output GPs fs.\n\njulia> ind_mogp1 = independent_mogp([GP(KernelFunctions.SEKernel())]);\n\njulia> ind_mogp2 = IndependentMOGP([GP(KernelFunctions.SEKernel())]);\n\njulia> typeof(ind_mogp1) == typeof(ind_mogp2)\ntrue\n\njulia> ind_mogp1.fs == ind_mogp2.fs\ntrue\n\n\n\n\n\n","category":"method"},{"location":"#LinearMixingModels.project-Union{Tuple{Z}, Tuple{AbstractMatrix{Z}, Z}} where Z<:Real","page":"Home","title":"LinearMixingModels.project","text":"project(H, σ²)\n\nComputes the projection T and ΣT given the mixing matrix and noise.\n\n\n\n\n\n","category":"method"},{"location":"#LinearMixingModels.project-Union{Tuple{Z}, Tuple{Orthogonal{Z, TU, TS} where {TU<:AbstractMatrix{Z}, TS<:(LinearAlgebra.Diagonal{Z, V} where V<:AbstractVector{Z})}, Z}} where Z<:Real","page":"Home","title":"LinearMixingModels.project","text":"project(H, σ²)\n\nComputes the projection T and ΣT given the mixing matrix and noise.\n\n\n\n\n\n","category":"method"},{"location":"#LinearMixingModels.regulariser-Tuple{Any, AbstractMatrix{var\"#s2\"} where var\"#s2\"<:Real}","page":"Home","title":"LinearMixingModels.regulariser","text":"regulariser(fx, y)\n\nComputes the regularisation term of the logpdf. See e.g. appendix A.4 of [1] - Bruinsma et al 2020.\n\n\n\n\n\n","category":"method"},{"location":"#LinearMixingModels.regulariser-Union{Tuple{T}, Tuple{Orthogonal{T, TU, TS} where {TU<:AbstractMatrix{T}, TS<:(LinearAlgebra.Diagonal{T, V} where V<:AbstractVector{T})}, T, AbstractMatrix{T}}} where T<:Real","page":"Home","title":"LinearMixingModels.regulariser","text":"regulariser(fx, y)\n\nComputes the regularisation term of the logpdf. See e.g. appendix A.4 of [1] - Bruinsma et al 2020.\n\n\n\n\n\n","category":"method"}]
}