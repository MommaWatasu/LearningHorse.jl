using LearningHorse
using Statistics
using Test

@testset "LearningHorse.jl" begin
    test_home_dir = pwd()
    include("./Preprocessing.jl")
    @info "Complete the test for Preprocessing"
    cd(test_home_dir)
    include("./LossFunction.jl")
    @info "Complete the test for LossFunction"
    include("./Regression.jl")
    @info "Complete the test for Regression"
    include("./Classification.jl")
    @info "Complete the test for Classification"
    include("./NeuralNetwork.jl")
    @info "Complete the test for NeuralNetwork"
end
