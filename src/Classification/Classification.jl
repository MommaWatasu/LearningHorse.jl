module Classification

using LinearAlgebra
export SVC, Logistic, LabelEncoder, OneHotEncoder

include("utils.jl")
include("Other.jl")
include("Logistic.jl")
include("SVC.jl")
end