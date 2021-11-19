using Documenter
#using LearningHorse.NeuralNetwork
include("/home/ubuntu/code/LearningHorse/src/Regression/Regression.jl")
#include("/home/ubuntu/code/LearningHorse/src/Regression/BFM.jl")
include("/home/ubuntu/code/LearningHorse/src/NeuralNetwork/NN.jl")
makedocs(;
    sitename="LearningHorse",
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Regression" => "Regression.md",
            "Classification" => "Classification.md",
            "Preprocessing" => "Preprocessing.md",
            "LossFunction" => "LossFunction.md",
            "NeuralNetwork" => "neuralnetwork.md",
        ],
        "日本語版マニュアル" => "Japanese.md"
    ]
)
