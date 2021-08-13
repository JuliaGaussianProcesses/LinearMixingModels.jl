@testset "orthogonal_matrix" begin
    U = rand(5, 3)
    S = Diagonal(rand(3))

    @test_throws ArgumentError Orthogonal(U, S)

    U, S, _ = svd(rand(5,3))
    H = Orthogonal(U, Diagonal(S))

    @test size(H) == size(U)
    @test H.S == Diagonal(S)
    @test H.U == U
    @test H == U*sqrt(Diagonal(S))
end
