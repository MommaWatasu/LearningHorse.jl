using LearningHorse
using LearningHorse.NeuralNetwork: NetWork, Dense, relu, tanh, Adam, add_layer!, train!, mse
using Test

@testset "LearningHorse.jl" begin
    include("./Preprocessing.jl")
    include("./LinearRegression.jl")
    include("./Classification.jl")
    include("./NeuralNetwork.jl")
end
