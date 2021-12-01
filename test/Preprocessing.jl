using LearningHorse.Preprocessing

@testset "Preprocessing" begin
    
    #generating data
    x = 5 .+ 25 .* rand(20)
    t = 170 .- 108 .* exp.(-0.2 .* x) .+ 4 .* rand(20)
    x1 = 23 .* (t ./ 100).^2 .+ 2 .* rand(20)
    x1 = hcat(x, x1)
    xt = x'

    #Standard Scaler
    @testset "Standard Scaler" begin
        scaler = Standard()
        for (data, dim) in zip([x, xt], [1, 2])
            @test_nowarn Preprocessing.fit!(scaler, data, dims=dim)
            @test_nowarn transform!(scaler, data, dims=dim)
            x2 = fit_transform!(scaler, data, dims=dim)
            @test inv_transform!(scaler, x2, dims=dim) == data
        end
        @test_throws DimensionMismatch transform!(scaler, x1)
        @test_nowarn fit_transform!(scaler, x1)
    end

    #MinMaxScaler
    @testset "MinMax Scaler" begin
        scaler = MinMax()
        for (data, dim) in zip([x, xt], [1, 2])
            @test_nowarn Preprocessing.fit!(scaler, data, dims=dim)
            @test_nowarn transform!(scaler, data, dims=dim)
            x2 = fit_transform!(scaler, data, dims=dim)
            @test inv_transform!(scaler, x2, dims=dim) == data
        end
        @test_throws DimensionMismatch transform!(scaler, x1)
        @test_nowarn x2 = fit_transform!(scaler, x1)
    end
    
    @testset "Robust Scaler" begin
        scaler = Robust()
        for (data, dim) in zip([x, xt], [1, 2])
            @test_nowarn Preprocessing.fit!(scaler, data, dims=dim)
            @test_nowarn transform!(scaler, data, dims=dim)
            x2 = fit_transform!(scaler, data, dims=dim)
            @test inv_transform!(scaler, x2, dims=dim) == data
        end
        @test_throws DimensionMismatch transform!(scaler, x1)
        @test_nowarn x2 = fit_transform!(scaler, x1)
    end
    
    #Data Preprocessor
    @testset "Data Preprocessing" begin
        originstdin = stdin
        f = open("data_test.txt")
        redirect_stdin(f)
        @test_nowarn dataloader("MNIST")
        @test_nowarn dataloader("BostonHousing")
        @test_nowarn dataloader("iris")
        @test_throws ArgumentError dataloader("dammy.csv")
        close(f)
    end
    
    @testset "Data Splitter" begin
        DS = DataSplitter(50, train_size = 0.3)
        data = rand(5, 50)
        @test (15, 5) == size(DS(data'))
        @test (5, 15) == size(DS(data, ndims = 2))
    end
end