using LearningHorse.NeuralNetwork

@testset "NeuralNetwork" begin
    #Test for activations
    data = [(rand(Float64, 10), rand(Float64)) for i in 1 : 10]
    loss = LossFunction.mse
    opt = Descent()
    NN = NetWork(Dense(10=>5, relu), Dense(5=>1, tanh))
    train!(NN, loss, data, opt)
    println("OK")
    @test_nowarn @epochs 10 train!(NN, loss, data, opt)
    @test typeof(NN[1]) <: Dense
    println(NN)
    @test_nowarn NN(rand(10, 10))
    
    #Test for optimizers
    for opt in [Descent(), Momentum(), AdaGrad(), Adam()]
        NN = NetWork(Dense(10=>5, relu), Dense(5=>1, tanh))
        @test_nowarn train!(NN, loss, data, opt)
    end
    
    #Test for Conv, Pooling, Flatten, Dropout
    data = [(rand(Float64, 5, 5, 2, 1), rand(Float64)) for i in 1 : 5]
    NN = NetWork(Conv((2, 2), 2=>1, relu), MaxPool((2, 2)), Flatten(), Dropout(0.25), Dense(4=>1, tanh))
    opt = Descent()
    @test_nowarn train!(NN, loss, data, opt)
    println(NN)
    @test_nowarn NN(rand(5, 5, 2, 1))
end