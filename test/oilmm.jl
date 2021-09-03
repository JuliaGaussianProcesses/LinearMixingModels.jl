function test_oilmm(rng, kernels, H::Orthogonal, x_train, x_test, y_train, y_test)
    fs = independent_mogp(map(GP, kernels))

    ilmm = ILMM(fs, collect(H))
    oilmm = ILMM(fs, H)

    ilmmx = ilmm(x_train, 0.1)
    oilmmx = oilmm(x_train, 0.1)

    @test isapprox(mean(ilmmx), mean(oilmmx))
    @test isapprox(var(ilmmx), var(oilmmx))
    @test isapprox(logpdf(ilmmx, y_train), logpdf(oilmmx, y_train))
    @test _is_approx(marginals(ilmmx), marginals(oilmmx))
    @test length(rand(rng, oilmmx)) == size(H, 1) * length(x_train.x)

    p_ilmmx = posterior(ilmmx, y_train)
    p_oilmmx = posterior(oilmmx, y_train)

    pi = p_ilmmx(x_test, 0.1)
    po = p_oilmmx(x_test, 0.1)

    @test isapprox(mean(pi), mean(po))
    @test isapprox(var(pi), var(po))
    @test isapprox(logpdf(pi, y_test), logpdf(po, y_test))
    @test _is_approx(marginals(pi), marginals(po))
    @test length(rand(rng, po)) == size(H, 1) * length(x_test.x)

    test_sampling_consistency(rng, oilmm, x_train)

    @test gradient(logpdf, oilmmx, y_train) isa Tuple
    @test gradient(logpdf, po, y_test) isa Tuple

    @testset "primary_public_interface" begin
        test_finitegp_primary_and_secondary_public_interface(rng, oilmmx)
        test_finitegp_primary_and_secondary_public_interface(rng, po)
    end
end

@testset "oilmm" begin
    rng = Random.seed!(04161999)
    x_train, x_test, y_train, y_test = generate_toy_data(rng)

    @testset "Full Rank, Dense H" begin
        U, S, _ = svd(rand(rng, 3, 3))
        H = Orthogonal(U, Diagonal(S))
        kernels = [SEKernel(), Matern32Kernel(), Matern32Kernel()]
        test_oilmm(rng, kernels, H, x_train, x_test, y_train, y_test)
    end

    @testset "M Latent Processes" begin
        U, S, _ = svd(rand(rng, 3, 2))
        H = Orthogonal(U, Diagonal(S))
        kernels = [SEKernel(), Matern32Kernel()]
        test_oilmm(rng, kernels, H, x_train, x_test, y_train, y_test)
    end

    @testset "1 Latent Processes" begin
        U, S, _ = svd(rand(rng, 3, 1))
        H = Orthogonal(U, Diagonal(S))
        kernels = [SEKernel()]
        test_oilmm(rng, kernels, H, x_train, x_test, y_train, y_test)
    end
end
@info "Ran oilmm tests."
