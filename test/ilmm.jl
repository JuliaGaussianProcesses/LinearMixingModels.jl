@testset "ILMM" begin
    rng = Random.seed!(04161999)
    x = range(0, 10; length=5)
    ys = rand(rng, GP(SEKernel())(x, 1e-6), 3)
    y1 = ys[:, 1]
    y2 = ys[:, 2]
    y3 = ys[:, 3]
    indices = randcycle(rng, 5)
    x_train = zeros(3)
    y_1_train = zeros(3)
    y_2_train = zeros(3)
    y_3_train = zeros(3)
    x_test = zeros(2)
    y_1_test = zeros(2)
    y_2_test = zeros(2)
    y_3_test = zeros(2)
    for (i, val) in enumerate(indices)
        if i <= 3
            x_train[i] = x[val]
            y_1_train[i] = y1[val]
            y_2_train[i] = y2[val]
            y_3_train[i] = y3[val]
        else
            x_test[i - 3] = x[val]
            y_1_test[i - 3] = y1[val]
            y_2_test[i - 3] = y2[val]
            y_3_test[i - 3] = y3[val]
        end
    end
    x_train = MOInputIsotopicByOutputs(x_train, 3)
    x_test = MOInputIsotopicByOutputs(x_test, 3)
    y_train = vcat(y_1_train, y_2_train, y_3_train)
    y_test = vcat(y_1_test, y_2_test, y_3_test)

    @testset "Full Rank, Dense H" begin
        H = rand(3, 3)
        k1, k2, k3 = SEKernel(), Matern32Kernel(), Matern32Kernel()

        ilmm = ILMM(independent_mogp([GP(k1), GP(k2), GP(k3)]), H)
        n_ilmm = GP(LinearMixingModelKernel([k1, k2, k3], H'))

        ilmmx = ilmm(x_train, 1e-6)
        n_ilmmx = n_ilmm(x_train, 1e-6)

        # @test isapprox(mean(ilmmx), mean(n_ilmmx))
        # @test isapprox(var(ilmmx), var(n_ilmmx))
        # @test isapprox(logpdf(ilmmx, y_train), logpdf(n_ilmmx, y_train))
        # @test _is_approx(marginals(ilmmx), marginals(n_ilmmx))

        @test Zygote.gradient(logpdf, ilmmx, y_train) isa Tuple
        @test Zygote.gradient(logpdf, n_ilmmx, y_train) isa Tuple

        p_ilmmx = posterior(ilmmx, y_train)
        p_n_ilmmx = posterior(n_ilmmx, y_train)

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

    @testset "M Latent Processes" begin
        H = rand(3,2)

        k1, k2 = SEKernel(), Matern32Kernel()

        ilmm = ILMM(independent_mogp([GP(k1), GP(k2)]), H)
        n_ilmm = GP(LinearMixingModelKernel([k1, k2], H'))

        ilmmx = ilmm(x_train, 1e-6)
        n_ilmmx = n_ilmm(x_train, 1e-6)

        # @test isapprox(mean(ilmmx), mean(n_ilmmx))
        # @test isapprox(var(ilmmx), var(n_ilmmx))
        # @test isapprox(logpdf(ilmmx, y_train), logpdf(n_ilmmx, y_train))
        # @test _is_approx(marginals(ilmmx), marginals(n_ilmmx))

        @test Zygote.gradient(logpdf, ilmmx, y_train) isa Tuple
        @test Zygote.gradient(logpdf, n_ilmmx, y_train) isa Tuple

        p_ilmmx = posterior(ilmmx, y_train)
        p_n_ilmmx = posterior(n_ilmmx, y_train)

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

    @testset "1 Latent Processes" begin
        H = rand(3,1)

        k = SEKernel()

        ilmm = ILMM(independent_mogp([GP(k)]), H)
        n_ilmm = GP(LinearMixingModelKernel([k], H'))

        ilmmx = ilmm(x_train, 1e-6)
        n_ilmmx = n_ilmm(x_train, 1e-6)

        # @test isapprox(mean(ilmmx), mean(n_ilmmx))
        # @test isapprox(var(ilmmx), var(n_ilmmx))
        # @test isapprox(logpdf(ilmmx, y_train), logpdf(n_ilmmx, y_train))
        # @test _is_approx(marginals(ilmmx), marginals(n_ilmmx))

        @test Zygote.gradient(logpdf, ilmmx, y_train) isa Tuple
        @test Zygote.gradient(logpdf, n_ilmmx, y_train) isa Tuple

        p_ilmmx = posterior(ilmmx, y_train)
        p_n_ilmmx = posterior(n_ilmmx, y_train)

        pi = p_ilmmx(x_test, 1e-6)
        pni = p_n_ilmmx(x_test, 1e-6)

        # @test isapprox(mean(pi), mean(pni))
        # @test isapprox(var(pi), var(pni))
        # @test isapprox(logpdf(pi, y_test), logpdf(pni, y_test))
        # @test _is_approx(marginals(pi), marginals(pni))

        @test Zygote.gradient(logpdf, pi, y_test) isa Tuple
        @test Zygote.gradient(logpdf, pni, y_test) isa Tuple

        # @testset "util" begin
        #     Σ = Diagonal(Fill(2, 3))
        #     @test noise_var(Σ) == 2

        #     y = rand(16)
        #     @test size(reshape_y(y, 8)) == (2, 8)
        #     @test size(reshape_y(y, 2)) == (8, 2)

        #     fs = independent_mogp([GP(Matern32Kernel())])
        #     H = rand(2, 1)
        #     x = MOInputIsotopicByOutputs(ColVecs(rand(2, 2)), 2)
        #     ilmm = ILMM(fs, H)
        #     ilmmx = ilmm(x, 0.1)
        #     @test (fs, H, 0.1, x.x) == unpack(ilmmx)

        #     @test get_latent_gp(ilmm) == fs
        # end

        # @testset "primary_public_interface" begin
        #     test_finitegp_primary_public_interface(rng, ilmmx)
        #     test_finitegp_primary_public_interface(rng, pi)
        # end
    end
end
