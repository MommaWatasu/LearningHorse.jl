using Documenter
using LearningHorse.Preprocessing
using LearningHorse.NeuralNetwork
using LearningHorse.Regression
using LearningHorse.LossFunction
using LearningHorse.Classification
using LearningHorse.Tree

makedocs(;
    sitename="LearningHorse",
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Regression" => "Regression.md",
            "Classification" => "Classification.md",
            "Preprocessing" => "Preprocessing.md",
            "LossFunction" => "LossFunction.md",
            "NeuralNetwork" => "NeuralNetwork.md",
        ],
        "日本語版マニュアル" => "Japanese.md"
    ]
)
