using LearningHorse.Preprocessing

@testset "Preprocessing" begin
    
    #generating data
    x = 5 .+ 25 .* rand(20)
    t = 170 .- 108 .* exp.(-0.2 .* x) .+ 4 .* rand(20)
    x1 = 23 .* (t ./ 100).^2 .+ 2 .* rand(20)
    x1 = hcat(x, x1)

    #Standard Scaler
    @testset "Standard Scaler" begin
        scaler = Standard()
        @test_nowarn Preprocessing.fit!(scaler, x)
        @test_nowarn transform!(scaler, x)
        x2 = fit_transform!(scaler, x)
        @test inv_transform!(scaler, x2) == x
        @test_throws DimensionMismatch transform!(scaler, x1)
        @test_nowarn fit_transform!(scaler, x1)
    end

    #MinMaxScaler
    @testset "MinMax Scaler" begin
        scaler = MinMax()
        @test_nowarn Preprocessing.fit!(scaler, x)
        @test_nowarn transform!(scaler, x)
        x2 = fit_transform!(scaler, x)
        @test inv_transform!(scaler, x2) == x
        @test_throws DimensionMismatch transform!(scaler, x1)
        @test_nowarn x2 = fit_transform!(scaler, x1)
    end
    
    @testset "Robust Scaler" begin
        scaler = Robust()
        @test_nowarn Preprocessing.fit!(scaler, x)
        @test_nowarn transform!(scaler, x)
        x2 = fit_transform!(scaler, x)
        @test inv_transform!(scaler, x2) == x
        @test_throws DimensionMismatch transform!(scaler, x1)
        @test_nowarn x2 = fit_transform!(scaler, x1)
    end
    
    ##TODO : add the test for Data.jl
    
end