function generate_toy_data(rng::AbstractRNG)
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

    return x_train, x_test, y_train, y_test
end

# Posterior should approximately agree with observations samples from the prior, if
# observed under a sufficiently small amount of noise.
# Testing for really eggregious bugs, as opposed to numerical issues, so tolerances are
# quite loose. 
function test_sampling_consistency(rng, f, x; rtol=1e-2, atol=1e2, σ²=1e-6)
    fx = f(x, σ²)
    y = rand(fx)
    f_post = posterior(fx, y)
    @test rand(rng, f_post(x, σ²)) ≈ y rtol = rtol
    @test mean(f_post(x)) ≈ y rtol = rtol
    @test var(f_post(x)) ≈ zeros(length(y)) rtol = rtol atol = atol
end

function approx_equivalent(rng::AbstractRNG, fx1::FiniteGP, fx2::FiniteGP)
    @test length(fx1) == length(fx2)

    @test mean(fx1) ≈ mean(fx2)
    @test var(fx1) ≈ var(fx2)
    @test cov(fx1) ≈ cov(fx2)

    y = rand(rng, fx1)
    @test logpdf(fx1, y) ≈ logpdf(fx2, y)
end
