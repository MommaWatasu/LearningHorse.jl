function bootstrap(x, t, n_trees)
    n_features = size(x)[2]
    n_features_forest = Int(floor(sqrt(n_features)))
    bootstrapped_x = zeros((length(t), n_features_forest, n_trees))
    bootstrapped_t = zeros(Int, n_trees, length(t))
    using_feature = zeros(Int, n_trees, n_features_forest)
    newaxis = [CartesianIndex()]
    global bootstrapped_x
    global bootstrapped_t
    global using_feature
    for i in 1 : n_trees
        ind = rand(1:length(t), length(t))
        col = sample(1 : n_features, n_features_forest, replace = false)
        k = x[ind, col]
        bootstrapped_x[:, :, i] = k[:, :, newaxis]
        bootstrapped_t[i, :] = t[ind]
        using_feature[i, :] = col
    end
    return bootstrapped_x, bootstrapped_t, using_feature
end

mutable struct RandomForest
    α::Float64
    n_trees::Int64
    forest::Array{DecisionTree, 1}
    classes::Array{Array{Any, 1}, 1}
    using_feature::Array{Int64, 2}
    RandomForest(nt::Int64; alpha::Float64 = 0.01) = new(alpha, nt, Array{DecisionTree}(undef, 0), Array{Array}(undef, 0))
end

function fit!(model::RandomForest, x, t)
    forest = Array{DecisionTree}(undef, model.n_trees)
    classes = Array{Array}(undef, model.n_trees)
    bootstrapped_x, bootstrapped_t, using_feature = bootstrap(x, t, model.n_trees)
    for i in 1:model.n_trees
        tree = DecisionTree(alpha = model.α)
        fit!(tree, bootstrapped_x[:, :, i], bootstrapped_t[i, :])
        forest[i], classes[i] = tree, tree.classes
    end
    model.forest = forest
    model.classes = classes
    model.using_feature = using_feature
end

function predict(model::RandomForest, x)
    solution = Array{Any}(undef, model.n_trees, size(x, 1))
    for (tree, feature, i) in zip(model.forest, model.using_feature, 1:model.n_trees)
        solution[i, :] = predict(tree, x[:, feature])
    end
    predicts = Array{Any}(undef, size(x, 1))
    for i in 1:size(x, 1)
        class, counts = unique(solution[:, i], count = true)
        predicts[i] = class[argmax(counts)]
    end
    return predicts
end