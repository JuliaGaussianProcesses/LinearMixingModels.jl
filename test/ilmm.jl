function test_ilmm(rng, kernels, H, x_train, x_test, y_train, y_test)

    # Construct ILMM and equivalent GP against which to test.
    ilmm = ILMM(independent_mogp(map(GP, kernels)), H)
    n_ilmm = GP(LinearMixingModelKernel(kernels, H'))

    ilmmx = ilmm(x_train, 1e-6)
    n_ilmmx = n_ilmm(x_train, 1e-6)

    # @test isapprox(mean(ilmmx), mean(n_ilmmx))
    # @test isapprox(var(ilmmx), var(n_ilmmx))
    # @test isapprox(logpdf(ilmmx, y_train), logpdf(n_ilmmx, y_train))
    # @test _is_approx(marginals(ilmmx), marginals(n_ilmmx))

    p_ilmmx = posterior(ilmmx, y_train)
    p_n_ilmmx = posterior(n_ilmmx, y_train)

    @test Zygote.gradient(logpdf, ilmmx, y_train) isa Tuple
    @test Zygote.gradient(logpdf, n_ilmmx, y_train) isa Tuple

    pi = p_ilmmx(x_test, 1e-6)
    pni = p_n_ilmmx(x_test, 1e-6)

    # @test isapprox(mean(pi), mean(pni))
    # @test isapprox(var(pi), var(pni))
    # @test isapprox(logpdf(pi, y_test), logpdf(pni, y_test))
    # @test _is_approx(marginals(pi), marginals(pni))

    @test Zygote.gradient(logpdf, pi, y_test) isa Tuple
    @test Zygote.gradient(logpdf, pni, y_test) isa Tuple

    # @testset "primary_public_interface" begin
    #     test_finitegp_primary_public_interface(rng, ilmmx)
    #     test_finitegp_primary_public_interface(rng, pi)
    # end
end

@testset "ilmm" begin
    rng = Random.seed!(04161999)
    x_train, x_test, y_train, y_test = generate_toy_data(rng)

    @testset "Full Rank, Dense H" begin
        H = rand(3, 3)
        kernels = [SEKernel(), Matern32Kernel(), Matern32Kernel()]
        test_ilmm(rng, kernels, H, x_train, x_test, y_train, y_test)
    end

    @testset "M Latent Processes" begin
        H = rand(3, 2)
        kernels = [SEKernel(), Matern32Kernel()]
        test_ilmm(rng, kernels, H, x_train, x_test, y_train, y_test)
        # @testset "util" begin
        #     Σ = Diagonal(Fill(2, 3))
        #     @test noise_var(Σ) == 2

        #     y = rand(16)
        #     @test size(reshape_y(y, 8)) == (2, 8)
        #     @test size(reshape_y(y, 2)) == (8, 2)

        #     @test get_latent_gp(ilmm) == independent_mogp(map(GP, kernels))
        # end
    end

    @testset "1 Latent Processes" begin
        H = rand(3, 1)
        kernels = [SEKernel()]
        test_ilmm(rng, kernels, H, x_train, x_test, y_train, y_test)
    end
end
@info "Ran ilmm tests."
