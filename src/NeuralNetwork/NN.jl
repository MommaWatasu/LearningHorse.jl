module NeuralNetwork

using LinearAlgebra
using NNlib
using Zygote
using Zygote: @adjoint

ACTIVATIONS = [
    :σ, :hardσ, :hardtanh, :relu,
    :leakyrelu, :relu6, :rrelu, :elu, :gelu, :swish, :selu,
    :celu, :softplus, :softsign, :logσ, :logcosh,
    :mish, :tanhshrink, :softshrink, :trelu, :lisht,
    :tanh_fast, :sigmoid_fast,
    ]

for f in ACTIVATIONS
    @eval export $(f)
end

export NetWork, Dense, Conv, Flatten, Dropout, MaxPool, add_layer!, train!, @epochs, params

include("utils.jl")
include("Layers/Network.jl")
include("Optimize/train.jl")

end