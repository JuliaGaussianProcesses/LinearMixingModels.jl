@testset "OILMM" begin
rng = Random.seed!(04161999)
    @testset "Full Rank, Dense H" begin
        U, S, _ = svd(rand(rng, 3, 3))
        H = Orthogonal(U, Diagonal(S));
        fs = IndependentMOGP([GP(SEKernel()), GP(Matern32Kernel()), GP(Matern32Kernel())])

        ilmm = ILMM(fs, collect(H))
        oilmm = ILMM(fs, H)

        x = range(0,10;length=5);
        ys = rand(rng, GP(SEKernel())(x, 1e-6), 3)
        y1 = ys[:,1]
        y2 = ys[:,2]
        y3 = ys[:,3]
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
            if i<=3
                x_train[i] = x[val]
                y_1_train[i] = y1[val]
                y_2_train[i] = y2[val]
            y_3_train[i] = y3[val]
            else
                x_test[i-3] = x[val]
                y_1_test[i-3] = y1[val]
                y_2_test[i-3] = y2[val]
            y_3_test[i-3] = y3[val]
            end
        end
        x_train = kf.MOInputIsotopicByOutputs(x_train, 3)
        x_test = kf.MOInputIsotopicByOutputs(x_test, 3)
        y_train = vcat(y_1_train, y_2_train, y_3_train)
        y_test = vcat(y_1_test, y_2_test, y_3_test);

        ilmmx = ilmm(x_train, 0.1)
        oilmmx = oilmm(x_train, 0.1)

        @test isapprox(mean(ilmmx), mean(n_ilmmx), atol=1e-4)
        @test isapprox(var(ilmmx), var(n_ilmmx), atol=1e-4)
        @test isapprox(logpdf(ilmmx, y_train), logpdf(n_ilmmx, y_train), atol=1e-4)
        @test marginals(ilmmx) == marginals(n_ilmmx)

        p_ilmmx = posterior(ilmmx, y_train);
        p_oilmmx = posterior(n_ilmmx, y_train);

        pi = p_ilmmx(x_test, 0.1);
        pni = p_n_ilmmx(x_test, 0.1);

        @test isapprox(mean(pi), mean(pni), atol=1e-4)
        @test isapprox(var(pi), var(pni), atol=1e-4)
        @test isapprox(logpdf(pi, y_test), logpdf(pni, y_test), atol=1e-4)
        @test marginals(pi) == marginals(pni)

        @testset "primary_public_interface" begin
            test_finitegp_primary_public_interface(rng, ilmmx)
            test_finitegp_primary_public_interface(rng, pi)
        end
    end

    @testset "M Latent Processes" begin
        U, S, _ = svd(rand(rng, 3, 2))
        H = Orthogonal(U, Diagonal(S));
        fs = IndependentMOGP([GP(SEKernel()), GP(Matern32Kernel())])
        ilmm = ILMM(fs, collect(H))
        oilmm = ILMM(fs, H)

        x = range(0,10;length=5);
        ys = rand(rng, GP(SEKernel())(x, 1e-6), 3)
        y1 = ys[:,1]
        y2 = ys[:,2]
        y3 = ys[:,3]
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
            if i<=3
                x_train[i] = x[val]
                y_1_train[i] = y1[val]
                y_2_train[i] = y2[val]
            y_3_train[i] = y3[val]
            else
                x_test[i-3] = x[val]
                y_1_test[i-3] = y1[val]
                y_2_test[i-3] = y2[val]
            y_3_test[i-3] = y3[val]
            end
        end
        x_train = kf.MOInputIsotopicByOutputs(x_train, 3)
        x_test = kf.MOInputIsotopicByOutputs(x_test, 3)
        y_train = vcat(y_1_train, y_2_train, y_3_train)
        y_test = vcat(y_1_test, y_2_test, y_3_test);

        ilmmx = ilmm(x_train, 0.1)
        n_ilmmx = n_ilmm(x_train, 0.1)

        @test isapprox(mean(ilmmx), mean(n_ilmmx), atol=1e-4)
        @test isapprox(var(ilmmx), var(n_ilmmx), atol=1e-4)
        @test isapprox(logpdf(ilmmx, y_train), logpdf(n_ilmmx, y_train), atol=1e-4)
        @test marginals(ilmmx) == marginals(n_ilmmx)

        p_ilmmx = posterior(ilmmx, y_train);
        p_n_ilmmx = posterior(n_ilmmx, y_train);

        pi = p_ilmmx(x_test, 0.1);
        pni = p_n_ilmmx(x_test, 0.1);

        @test isapprox(mean(pi), mean(pni), atol=1e-4)
        @test isapprox(var(pi), var(pni), atol=1e-4)
        @test isapprox(logpdf(pi, y_test), logpdf(pni, y_test), atol=1e-4)
        @test marginals(pi) == marginals(pni)

        @testset "primary_public_interface" begin
            test_finitegp_primary_public_interface(rng, ilmmx)
            test_finitegp_primary_public_interface(rng, pi)
        end
    end

    @testset "1 Latent Processes" begin
        U, S, _ = svd(rand(rng, 3, 1))
        H = Orthogonal(U, Diagonal(S));
        fs = IndependentMOGP([GP(SEKernel())])

        ilmm = ILMM(fs, collect(H))
        oilmm = ILMM(fs, H)

        x = range(0,10;length=5);
        ys = rand(rng, GP(SEKernel())(x, 1e-6), 3)
        y1 = ys[:,1]
        y2 = ys[:,2]
        y3 = ys[:,3]
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
            if i<=3
                x_train[i] = x[val]
                y_1_train[i] = y1[val]
                y_2_train[i] = y2[val]
            y_3_train[i] = y3[val]
            else
                x_test[i-3] = x[val]
                y_1_test[i-3] = y1[val]
                y_2_test[i-3] = y2[val]
            y_3_test[i-3] = y3[val]
            end
        end
        x_train = kf.MOInputIsotopicByOutputs(x_train, 3)
        x_test = kf.MOInputIsotopicByOutputs(x_test, 3)
        y_train = vcat(y_1_train, y_2_train, y_3_train)
        y_test = vcat(y_1_test, y_2_test, y_3_test);

        ilmmx = ilmm(x_train, 0.1)
        n_ilmmx = n_ilmm(x_train, 0.1)

        @test isapprox(mean(ilmmx), mean(n_ilmmx), atol=1e-4)
        @test isapprox(var(ilmmx), var(n_ilmmx), atol=1e-4)
        @test isapprox(logpdf(ilmmx, y_train), logpdf(n_ilmmx, y_train), atol=1e-4)
        @test marginals(ilmmx) == marginals(n_ilmmx)

        p_ilmmx = posterior(ilmmx, y_train);
        p_n_ilmmx = posterior(n_ilmmx, y_train);

        pi = p_ilmmx(x_test, 0.1);
        pni = p_n_ilmmx(x_test, 0.1);

        @test isapprox(mean(pi), mean(pni), atol=1e-4)
        @test isapprox(var(pi), var(pni), atol=1e-4)
        @test isapprox(logpdf(pi, y_test), logpdf(pni, y_test), atol=1e-4)
        @test marginals(pi) == marginals(pni)

        @testset "primary_public_interface" begin
            test_finitegp_primary_public_interface(rng, ilmmx)
            test_finitegp_primary_public_interface(rng, pi)
        end
    end
end
