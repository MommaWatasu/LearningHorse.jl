# LearningHorse

![logo](https://user-images.githubusercontent.com/76277264/105711246-a16efc00-5f5b-11eb-9e1e-e93955250bd8.png)

|Jenkins|GitHub Actions|Coveralls|CodeCov|
|:-----:|:------------:|:-----:|:------:|
|[![Build Status](https://ik1-414-39231.vs.sakura.ne.jp:8443/buildStatus/icon?job=LearningHorseCI)](https://ik1-414-39231.vs.sakura.ne.jp:8443/job/LearningHorseCI/)|[![CI](https://github.com/QGMW22/LearningHorse.jl/actions/workflows/main.yml/badge.svg)](https://github.com/QGMW22/LearningHorse.jl/actions/workflows/main.yml) [![CI-on-latest-julia](https://github.com/QGMW22/LearningHorse.jl/actions/workflows/on-nightly.yml/badge.svg)](https://github.com/QGMW22/LearningHorse.jl/actions/workflows/on-nightly.yml)|[![Coverage Status](http://coveralls.io/repos/github/QGMW22/LearningHorse.jl/badge.svg?branch=master)](https://coveralls.io/github/QGMW22/LearningHorse.jl?branch=master)|[![Coverage](https://codecov.io/gh/QGMW22/LearningHorse.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/QGMW22/LearningHorse.jl)|

LearningHorse is machine learning library for JuliaLang. You can use various algorithms, from basic algorithms such as linear regression to neuralnetwork.

## Resources
- Documentation:[LearningHorse docs](https://qgmw22.github.io/LearningHorse.jl/docs)
- Source code:[LearningHorse.jl (This repositry)](https://github.com/QGMW22/Learninghorse.jl)

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

## Improvement
ver 0.4.0 will be the first stable release.
