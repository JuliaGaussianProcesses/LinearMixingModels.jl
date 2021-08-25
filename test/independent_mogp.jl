@testset "independent_mogp" begin
    rng = Random.seed!(123)
    x = range(1, 2; length=5)
    ϵ = rand(Normal(0, 0.5), 5)
    y_1 = 30 .+ sqrt.(x) .* sin.(x) .+ ϵ
    y_2 = 10 .+ cbrt.(x) .* cos.(2x) .+ ϵ

    indices = randcycle(rng, 5)
    x_train = zeros(3)
    y_1_train = zeros(3)
    y_2_train = zeros(3)
    x_test = zeros(2)
    y_1_test = zeros(2)
    y_2_test = zeros(2)
    for (i, val) in enumerate(indices)
        if i<=3
            x_train[i] = x[val]
            y_1_train[i] = y_1[val]
            y_2_train[i] = y_2[val]
        else
            x_test[i-3] = x[val]
            y_1_test[i-3] = y_1[val]
            y_2_test[i-3] = y_2[val]
        end
    end

    x_train_mo = MOInputIsotopicByOutputs(x_train,2)
    x_test_mo = MOInputIsotopicByOutputs(x_test,2)
    y_train = vcat(y_1_train, y_2_train)
    y_test = vcat(y_1_test, y_2_test)


    f1 = GP(30, Matern32Kernel())
    f2 = GP(10, SEKernel())
    f = independent_mogp([f1, f2])
    fx = f(x_train_mo, 0.1)
    fx1 = f1(x_train, 0.1)
    fx2 = f2(x_train, 0.1)

    @test isapprox(
        logpdf(fx, y_train), logpdf(fx1, y_1_train) + logpdf(fx2, y_2_train)
    )
    @test marginals(fx) == vcat(marginals(fx1), marginals(fx2))
    @test isapprox(mean(fx), vcat(mean(fx1), mean(fx2)))
    @test isapprox(var(fx), vcat(var(fx1), var(fx2)))
    @test length(rand(rng, fx)) == length(f.fs) * length(x_train_mo.x)

    pfx = posterior(fx, y_train)
    pfx1 = posterior(fx1, y_1_train)
    pfx2 = posterior(fx2, y_2_train)
    post_fx = pfx(x_test_mo, 0.1)
    post_fx1 = pfx1(x_test, 0.1)
    post_fx2 = pfx2(x_test, 0.1)

    @test isapprox(
        logpdf(post_fx, y_test), logpdf(post_fx1, y_1_test) + logpdf(post_fx2, y_2_test)
    )
    @test marginals(post_fx) == vcat(marginals(post_fx1), marginals(post_fx2))
    @test isapprox(mean(post_fx), vcat(mean(post_fx1), mean(post_fx2)))
    @test isapprox(var(post_fx), vcat(var(post_fx1), var(post_fx2)))
    @test length(rand(rng, post_fx)) == length(f.fs) * length(x_test_mo.x)

    test_sampling_consistency(rng, f, x_train_mo)

    @testset "primary_public_interface" begin
        test_finitegp_primary_public_interface(rng, fx)
        test_finitegp_primary_public_interface(rng, post_fx)

        A = randn(rng, length(x_train_mo), length(x_train_mo))
        Σy = A'A + I
        test_finitegp_primary_and_secondary_public_interface(rng, f(x_train_mo, Σy))
        test_internal_abstractgps_interface(rng, f, x_train_mo, x_test_mo)
    end
end

@info "Ran independent_mogp tests."
