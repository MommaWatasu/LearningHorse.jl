using LearningHorse
using Test

@testset "LearningHorse.jl" begin
    mean = Loss_Function.MSE([1,2,3,4], [1,2,3,4], [1,2])
    @test mean == 4
    x = [15.43 23.01 5.0 12.56 8.67 7.31 9.66 13.64 14.92 18.47 15.48 22.13 10.11 26.95 5.68 21.76]
    t = [170.91 160.68 129.0 159.7 155.46 140.56 153.65 159.43 164.7 169.65 160.71 173.29 159.31 171.52 138.96 165.87]
    w =Linear_Regression.SGD.fit(x, t)
    p = Linear_Regression.SGD.predict(x, w)
    println(typeof(p), p)
    @test 1 < w[1] < 2
    @test 100 < w[2]
    x = [15.43 23.01 5.0 12.56 8.67 7.31 9.66 13.64 14.92 18.47 15.48 22.13 10.11 26.95 5.68 21.76]
    t = [170.91 160.68 129.0 159.7 155.46 140.56 153.65 159.43 164.7 169.65 160.71 173.29 159.31 171.52 138.96]
    w =Linear_Regression.SGD.fit(x, t)
    x =[15.43 23.01 5.0 12.56 8.67 7.31 9.66 13.64 14.92 18.47 15.48 22.13 10.11 26.95 5.68 21.76; 70.43 58.15 37.22 56.51 57.32 40.84 57.79 56.94 63.03 65.69 62.33 64.95 57.73 66.89 46.68 61.08]
    t = [170.91 160.68 129.0 159.7 155.46 140.56 153.65 159.43 164.7 169.65 160.71 173.29 159.31 171.52 138.96 165.87]
    w = Linear_Regression.MR.fit(x, t)
    println(typeof(w))
    p = Linear_Regression.MR.predict(x, w)
    print(p)
    w = Linear_Regression.LR.fit(x, t)
    println(w)
    p = Linear_Regression.LR.predict(x, w)
    println(typeof(p))
    println(p)

    # Write your tests here.
end
