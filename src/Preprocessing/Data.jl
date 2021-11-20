function dataloader(name; header = true, dir = "learningdatasets", tt = "both")
    hd = homedir()
    dir = joinpath(hd, dir)
    mkpath(dir)
    cd(dir)
    if name == "MNIST"
        if !(isfile("mnist_train.csv"))
            yn = Base.prompt("Can I download the mnist_train.csv(110MB)? (y/n)")
            yn == "y" && Downloads.download("https://pjreddie.com/media/files/mnist_train.csv", "mnist_train.csv")
        end
        df_r = CSV.read("mnist_train.csv", header = false, DataFrame)
        if !(isfile("mnist_test.csv"))
            yn = Base.prompt("Can I download the mnist_test.csv(18MB)? (y/n)")
            yn == "y" && Downloads.download("https://pjreddie.com/media/files/mnist_test.csv", "mnist_test.csv")
        end
        df_e = CSV.read("mnist_test.csv",header = false , DataFrame)
        return df_r, df_e
    elseif name == "iris"
        url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
        if !(isfile("iris.csv"))
            yn = Base.prompt("Can I download the iris.csv(4KB)? (y/n)")
            yn == "y" && Downloads.download(url, "iris.csv")
        end
        df = CSV.read("iris.csv", DataFrame)
        return df
    elseif name == "BostonHousing"
        url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
        if !(isfile("BostonHousing.csv"))
            yn = Base.prompt("Can I download the BostonHousing.csv(36KB)? (y/n)")
            yn == "y" && Downloads.download(url, "BostonHousing.csv")
        end
        df = CSV.read("BostonHousing.csv", DataFrame)
        return df
    end
    df = CSV.read(name, header = header, DataFrame)
    return df
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