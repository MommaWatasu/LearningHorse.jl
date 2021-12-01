module Tree

#using StatsBase
include("../utils.jl")

export DecisionTree, MV, RandomForest

include("utils.jl")
include("DT.jl")
include("MV.jl")
include("RF.jl")

end