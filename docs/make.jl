using Documenter
makedocs(;
    sitename="LearningHorse"
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Regression" => "Regression.md"
            "Classification" => "Classification.md"
            "Preprocessing" => "Preprocessing.md"
            "LossFunction" => "LossFunction.md"
        ],
        "日本語版マニュアル" => "japanese.md"
)
