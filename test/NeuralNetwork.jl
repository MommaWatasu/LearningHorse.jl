using LearningHorse.NeuralNetwork

@testset "NeuralNetwork" begin
    for func in [line, sigmoid, hardsigmoid, relu, relu6, elu, selu, tanh, hardtanh, softsign, softplus, mish, swish]
        NN = NetWork(Dense(10=>5, relu), Dense(5=>1, func))
        data = [(rand(Float64, 10), rand(Float64)) for i in 1 : 10]
        loss = LossFunction.mse
        opt = Adam()
        @test_nowarn train!(NN, loss, data, opt)
    end
end