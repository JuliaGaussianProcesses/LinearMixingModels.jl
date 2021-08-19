@testset "OILMM" begin
    rng = Random.seed!(04161999)
    atol = 1e-2
    @testset "Full Rank, Dense H" begin
        U, S, _ = svd(rand(rng, 3, 3))
        H = Orthogonal(U, Diagonal(S));
        fs = independent_mogp([GP(SEKernel()), GP(Matern32Kernel()), GP(Matern32Kernel())])

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
        x_train = MOInputIsotopicByOutputs(x_train, 3)
        x_test = MOInputIsotopicByOutputs(x_test, 3)
        y_train = vcat(y_1_train, y_2_train, y_3_train)
        y_test = vcat(y_1_test, y_2_test, y_3_test);

        ilmmx = ilmm(x_train, 0.1)
        oilmmx = oilmm(x_train, 0.1)

        @test isapprox(mean(ilmmx), mean(oilmmx), atol=atol)
        @test isapprox(var(ilmmx), var(oilmmx), atol=atol)
        @test isapprox(logpdf(ilmmx, y_train), logpdf(oilmmx, y_train), atol=atol)
        @test _is_approx(marginals(ilmmx), marginals(oilmmx))

        p_ilmmx = posterior(ilmmx, y_train);
        p_oilmmx = posterior(oilmmx, y_train);

        pi = p_ilmmx(x_test, 0.1);
        po = p_oilmmx(x_test, 0.1);

        @test isapprox(mean(pi), mean(po), atol=atol)
        @test isapprox(var(pi), var(po), atol=atol)
        @test isapprox(logpdf(pi, y_test), logpdf(po, y_test), atol=atol)
        @test _is_approx(marginals(pi), marginals(po))

        @testset "primary_public_interface" begin
            test_finitegp_primary_public_interface(rng, oilmmx)
            test_finitegp_primary_public_interface(rng, po)
        end
    end

    @testset "M Latent Processes" begin
        U, S, _ = svd(rand(rng, 3, 2))
        H = Orthogonal(U, Diagonal(S));
        fs = independent_mogp([GP(SEKernel()), GP(Matern32Kernel())])
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
        x_train = MOInputIsotopicByOutputs(x_train, 3)
        x_test = MOInputIsotopicByOutputs(x_test, 3)
        y_train = vcat(y_1_train, y_2_train, y_3_train)
        y_test = vcat(y_1_test, y_2_test, y_3_test);

        ilmmx = ilmm(x_train, 0.1)
        oilmmx = oilmm(x_train, 0.1)

        @test isapprox(mean(ilmmx), mean(oilmmx), atol=atol)
        @test isapprox(var(ilmmx), var(oilmmx), atol=atol)
        @test isapprox(logpdf(ilmmx, y_train), logpdf(oilmmx, y_train), atol=atol)
        @test _is_approx(marginals(ilmmx), marginals(oilmmx))

        p_ilmmx = posterior(ilmmx, y_train);
        p_oilmmx = posterior(oilmmx, y_train);

        pi = p_ilmmx(x_test, 0.1);
        po = p_oilmmx(x_test, 0.1);

        @test isapprox(mean(pi), mean(po), atol=atol)
        @test isapprox(var(pi), var(po), atol=atol)
        @test isapprox(logpdf(pi, y_test), logpdf(po, y_test), atol=atol)
        @test _is_approx(marginals(pi), marginals(po))

        @testset "primary_public_interface" begin
            test_finitegp_primary_public_interface(rng, oilmmx)
            test_finitegp_primary_public_interface(rng, po)
        end
    end

    @testset "1 Latent Processes" begin
        U, S, _ = svd(rand(rng, 3, 1))
        H = Orthogonal(U, Diagonal(S));
        fs = independent_mogp([GP(SEKernel())])

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
        x_train = MOInputIsotopicByOutputs(x_train, 3)
        x_test = MOInputIsotopicByOutputs(x_test, 3)
        y_train = vcat(y_1_train, y_2_train, y_3_train)
        y_test = vcat(y_1_test, y_2_test, y_3_test);

        ilmmx = ilmm(x_train, 0.1)
        oilmmx = oilmm(x_train, 0.1)

        @test isapprox(mean(ilmmx), mean(oilmmx), atol=atol)
        @test isapprox(var(ilmmx), var(oilmmx), atol=atol)
        @test isapprox(logpdf(ilmmx, y_train), logpdf(oilmmx, y_train), atol=atol)
        @test _is_approx(marginals(ilmmx), marginals(oilmmx))

        p_ilmmx = posterior(ilmmx, y_train);
        p_oilmmx = posterior(oilmmx, y_train);

        pi = p_ilmmx(x_test, 0.1);
        po = p_oilmmx(x_test, 0.1);

        @test isapprox(mean(pi), mean(po), atol=atol)
        @test isapprox(var(pi), var(po), atol=atol)
        @test isapprox(logpdf(pi, y_test), logpdf(po, y_test), atol=atol)
        @test _is_approx(marginals(pi), marginals(po))

        @testset "primary_public_interface" begin
            test_finitegp_primary_public_interface(rng, oilmmx)
            test_finitegp_primary_public_interface(rng, po)
        end
    end
end
