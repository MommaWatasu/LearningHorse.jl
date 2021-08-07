using MLDatasets
using Random

function dataloader(dataname::String)
    if dataname == "MNIST" || dataname == "mnist"
        train_x, train_y = MNIST.traindata()
        test_x, test_y = MNIST.testdata()
        train_x, train_y = collect(Float64, train_x), collect(Float64, train_y)
        test_x, test_y = collect(Float64, test_x), collect(Float64, test_y)
        return train_x, train_y, test_x, test_y
    elseif dataname == "iris"
        return Iris.features(), Iris.labels()
    elseif dataname == "BostonHousing"
        return BostonHousing.features(), BostonHousing.targets()
    end
end

struct DataSplitter
    #TODO: make it possible to DataFrame split
    splitted_data::Tuple
    function DataSplitter(xs, narray, ndata, sb::Bool, seed, test_size::Int)
        if sb == true
            seed != nothing && Random.seed!(seed)
            narray = shuffle(narray)
        end
        narray = (narray[1:test_size-1], narray[test_size:end])
        splitted_data = []
        for x in xs
            push!(splitted_data, index2element(x, narray))
        end
        new(Tuple(splitted_data))
    end
end

function index2element(x, index::Tuple)
    tr, te = index
    if ndims(x) == 1
        return ([x[i] for i in tr], [x[i] for i in te])
    else
        return ([x[i, :] for i in tr], [x[i, :] for i in te])
    end
end

#ndata is dims = 1
function split_data(x...; sb = true, train_size = nothing, test_size = nothing, seed = nothing)
    x = collect(x)
    ndata = size(x[1])[1]
    narray = collect(1 : 1 : ndata)
    test_size == train_size == nothing && throw(ArgumentError("test_size and train_size wasn't specified. you must specify either one."))
    if test_size == nothing
        #number of train data is ceiled
        test_size = (0<=train_size<=1) ? ndata-ceil(train_size*ndata) : ndata-train_size
    else
        test_size = (0<=test_size<=1) ? ndata-ceil(train_size*ndata) : ndata-train_size
    end
    return DataSplitter(x, narray, ndata, sb, seed, Int(test_size)).splitted_data
end