module NeuralNetwork

using LinearAlgebra
import Zygote

export NetWork, Dense, Conv, Flatten, Dropout, MaxPool, add_layer!, train!, epochs

include("./utils.jl")
include("./Dims.jl")
include("./Layers/Network.jl")
include("./Optimize/train.jl")

end