using LinearAlgebra
using Statistics
module LearningHorse

    export LinearRegression
    module LinearRegression
        module SGD() #this is Slope Gradient Descent
            include("./LinearRegression/SGD.jl")
        end
        module MR#this is Multiple Regression
            include("./LinearRegression/MR.jl")
        end
        module RR()#this is Ridge Regression
            include("./LinearRegression/RR.jl")
        end
        module LR()#this is Lasso Regression
            include("./LinearRegression/LR.jl")
        end
        module BFM() #this use Basis Function Model for regression
            include("./LinearRegression/BFM.jl")
        end
    end

    export Classification
    module Classification #this is Classification
        include("./Classification/Other.jl") #This file defined LE and CTOH
        module MS #This is Multiclass softmax
            include("./Classification/MS.jl")
        end
        module OVR #this is One-vs-Rest
            include("./Classification/OVR.jl")
        end
    end
    export Tree
    module Tree #This is Decision Tree
        module DT #This is Decision Tree
            include("./Tree/DT.jl")
        end
        include("./Tree/MV.jl")
    end

    export Ensemble
    module Ensemble
        module RF
            include("./Ensemble/RF.jl")
        end
    end
    export Preprocessing
    module Preprocessing
        module SS() #this is Standard Scaler
            include("./Preprocessing/SS.jl")
        end
        module MM() #this is MinMax Scaler
            include("./Preprocessing/MM.jl")
        end
        module RS() #this is Robust Scaler
            include("./Preprocessing/RS.jl")
        end
    end
    export LossFunction
    module LossFunction #Loss Fuunctions 
        include("./LossFunction/LossFunctions.jl")
    end

end
