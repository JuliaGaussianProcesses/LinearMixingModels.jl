@testset "independent_mogp" begin
    rng = Random.seed!(123);
    x = range(1,31;length=40)
    ϵ = rand(Normal(0,0.5), 40)
    y_1 = 30 .+ sqrt.(x).*sin.(x) .+ ϵ
    y_2 = 10 .+ cbrt.(x).*cos.(2*x) .+ ϵ

    indices =  randcycle(rng, 40)
    x_train = zeros(30)
    y_1_train = zeros(30)
    y_2_train = zeros(30)
    x_test = zeros(10)
    y_1_test = zeros(10)
    y_2_test = zeros(10)
    for (i, val) in enumerate(indices)
        if i<=30
            x_train[i] = x[val]
            y_1_train[i] = y_1[val]
            y_2_train[i] = y_2[val]
        else
            x_test[i-30] = x[val]
            y_1_test[i-30] = y_1[val]
            y_2_test[i-30] = y_2[val]
        end
    end

    x_train_mo = MOInputIsotopicByFeatures(x_train,2)
    x_test_mo = MOInputIsotopicByFeatures(x_test,2)
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
end
