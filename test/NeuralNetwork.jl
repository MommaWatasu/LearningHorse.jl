using LearningHorse.NeuralNetwork

@testset "NeuralNetwork" begin
    NN = NetWork(Dense(10=>5, relu), Dense(5=>1, tanh))
    data = [(rand(Float64, 10), rand(Float64)) for i in 1 : 10]
    loss = LossFunction.mse
    opt = Adam()
    @test_nowarn train!(NN, loss, data, opt)
end