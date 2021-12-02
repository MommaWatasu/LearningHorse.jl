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
            "Regression" => "Manual/Regression.md",
            "Classification" => "Manual/Classification.md",
            "Preprocessing" => "Manual/Preprocessing.md",
            "LossFunction" => "Manual/LossFunction.md",
            "NeuralNetwork" => "Manual/NeuralNetwork.md",
        ],
        "Tutorial" => [
            "Welcome to LearningHorse" => "Tutorial/Welcome.md",
            "Getting Started" => "Tutorial/Getting_Started.md",
        ],
    ]
)