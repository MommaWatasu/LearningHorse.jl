module Preprocessing

export Standard, MinMax, Robust, transform!, fit_transform!, inv_transform!

using Statistics
using Random
using CSV
using DataFrames

include("utils.jl")
include("Data.jl")
include("MM.jl")
include("RS.jl")
include("SS.jl")

end