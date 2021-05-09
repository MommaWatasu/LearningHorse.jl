@testset "LinearRegression" begin
    mean = LossFunction.MSE([1,2,3,4], [1,2,3,4], [1,2])
    @test mean == 4
    x = [15.43 23.01 5.0 12.56 8.67 7.31 9.66 13.64 14.92 18.47 15.48 22.13 10.11 26.95 5.68 21.76]
    t = [170.91 160.68 129.0 159.7 155.46 140.56 153.65 159.43 164.7 169.65 160.71 173.29 159.31 171.52 138.96 165.87]
    w =LinearRegression.SGD.fit(x, t)
    p = LinearRegression.SGD.predict(x, w)
    @test 1 < w[1] < 2
    @test 100 < w[2]
    x =[15.43 23.01 5.0 12.56 8.67 7.31 9.66 13.64 14.92 18.47 15.48 22.13 10.11 26.95 5.68 21.76; 70.43 58.15 37.22 56.51 57.32 40.84 57.79 56.94 63.03 65.69 62.33 64.95 57.73 66.89 46.68 61.08]
    t = [170.91 160.68 129.0 159.7 155.46 140.56 153.65 159.43 164.7 169.65 160.71 173.29 159.31 171.52 138.96 165.87]
    w = LinearRegression.MR.fit(x, t)
    p = LinearRegression.MR.predict(x, w)
    print(p)
    w = LinearRegression.RR.fit(x, t)
    p = LinearRegression.RR.predict(x, w)
    w = LinearRegression.LR.fit(x, t)
    println(w)
    p = LinearRegression.LR.predict(x, w)
    println(p)
    w, m, s = LinearRegression.BFM.fit(x[1, :], t, 4)
    println("w:", w, "m:", m, "s:", s)
    p = LinearRegression.BFM.predict(x[1, :], w, m, s)
    println(p)
    mse = LossFunction.MSE(x[1, :], t, w, b = "gauss", m = m, s = s)
    println(mse)
    w, m, s = LinearRegression.BFM.fit(x[1, :], t, 4, alpha = 0.1, f = "polynomial")
    println("p(polynomial):", p)
    p = LinearRegression.BFM.predict(x[1, :], w, m, s, f = "polynomial")
    println("p:", p)
    mse = LossFunction.MSE(x[1, :], t, w, b = "polynomial", m = m, s = s)
end