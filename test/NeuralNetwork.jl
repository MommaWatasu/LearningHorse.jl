@testset "NeuralNetwork" begin
    @test_nowarn begin
        NN = NetWork()
        add_layer!(NN, Dense(10=>5, relu))
        add_layer!(NN, Dense(5=>1, tanh))
        data = []
        for i in 1 : 10
            push!(data, (rand(Float64, 10), rand(Float64, 1)))
        end
        loss = mse
        opt = Adam()
        train!(NN, loss, data, opt)
    end
end