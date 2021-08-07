using Documenter
using LearningHorse.NeuralNetwork
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
