# LearningHorse

# Note: I'm sorry, this repository has been moved to [HorseML.jl](https://github.com/MommaWatasu/HorseML.jl) due to my account name change.

![logo](https://user-images.githubusercontent.com/76277264/105711246-a16efc00-5f5b-11eb-9e1e-e93955250bd8.png)

|GitHub Actions|Coveralls|CodeCov|Docs|
|:------------:|:-----:|:------:|:--------:|
|[![CI][CI-img]][CI-url] [![CI-on-nightly-julia][on-nightly-img]][on-nightly-url]|[![Coverage Status][coveralls-img]][coveralls-url]|[![codecov][codecov-img]][codecov-url]|[![docs dev][docs-dev-img]][docs-dev-url] [![docs stable][docs-stable-img]][docs-stable-url]|

LearningHorse is machine learning library for JuliaLang. You can use various algorithms, from basic algorithms such as linear regression to neuralnetwork.

## Installation
From the Julia REPL, type ] to enter the Pkg REPL mode and run.
```@example
pkg> add LearningHorse
```

## Abstract
This library is ML library for JuliaLang. LearningHorse can be installed using the Julia package manager.　You can build many models, from simple models like regression to complex models like neural networks.Therefore, this package is recommended for beginners.

## Example
With Learning Horse, you can create the following models.

This model is the Decision Tree, you can make the model visualized like this:
![tree](https://user-images.githubusercontent.com/76277264/144227810-035b0f07-6242-4cf3-9280-af9df17a6d6e.png)


This model is the polynomial Regression:

![polynomial](https://user-images.githubusercontent.com/76277264/144227907-c8f86ba4-afae-416a-ac30-a8ad1eff149c.png)


[CI-img]: https://github.com/MommaWatasu/LearningHorse.jl/actions/workflows/CI.yml/badge.svg
[CI-url]: https://github.com/MommaWatasu/LearningHorse.jl/actions/workflows/CI.yml

[on-nightly-img]: https://github.com/MommaWatasu/LearningHorse.jl/actions/workflows/on-nightly.yml/badge.svg
[on-nightly-url]: https://github.com/MommaWatasu/LearningHorse.jl/actions/workflows/on-nightly.yml

[coveralls-img]: https://coveralls.io/repos/github/MommaWatasu/LearningHorse.jl/badge.svg?branch=master
[coveralls-url]: https://coveralls.io/github/MommaWatasu/LearningHorse.jl?branch=master

[codecov-img]: https://codecov.io/gh/MommaWatasu/LearningHorse.jl/branch/master/graph/badge.svg?token=x7YKbXMjPE
[codecov-url]: https://codecov.io/gh/MommaWatasu/LearningHorse.jl

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://mommawatasu.github.io/LearningHorse.jl/dev

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://mommawatasu.github.io/LearningHorse.jl/dev