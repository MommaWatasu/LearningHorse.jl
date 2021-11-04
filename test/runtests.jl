using LearningHorse
using Statistics
using Test

@testset "LearningHorse.jl" begin
    include("./Preprocessing.jl")
    @info "Complete the test for Preprocessing"
    include("./LossFunction.jl")
    @info "Complete the test for LossFunction"
    include("./Regression.jl")
    @info "Complete the test for Regression"
    include("./Classification.jl")
    @info "Complete the test for Classification"
    include("./NeuralNetwork.jl")
    @info "Complete the test for NeuralNetwork"
end
