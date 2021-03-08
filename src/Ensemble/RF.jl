using StatsBase
function unique(x; count = false)
    s = [v for v in Set(x)]
    if count
        c = zeros(1, size(s)[1])
        for i in 1:length(x)
            for j in 1:length(s)
                if x[i] == s[j]
                    c[j] += 1
                    break
                end
            end
        end
        k = zeros(Int, size(c))
        for i in 1:length(c)
            k[i] = Int(c[i])
        end
        return s, k
    end
    return s
end
function DecisionTree(x, t; alpha = 0.01)
    function get_branching_shold(xs)
        u = unique(xs)
        return (u[2:length(u)] + u[1:length(u)-1]) / 2
    end
    function delta_gini_index(left, right, alpha, branch_count)
        n_left = length(left)
        n_right = length(right)
        n_total = n_left + n_right
        _, counts = unique(left, count = true)
        ratio_classes = counts / n_left
        left_gain = (n_left / n_total) * (1 .- sum(ratio_classes .^ 2))
        _, counts = unique(right, count = true)
        ratio_classes = counts / n_right
        right_gain = (n_right / n_total) * (1 .- sum(ratio_classes .^ 2))
        return left_gain + right_gain + alpha * branch_count
    end
    function branch(x, t, alpha, branch_count)
        gains = []
        rules = []
        for feature_id in 1 : size(x)[2]
            xs = x[:, feature_id]
            thresholds = get_branching_shold(xs)
            for th in thresholds
                f = findall(x -> x < th, xs)
                left_y = [t[i] for i in f]
                f = findall(x -> x >= th, xs)
                right_y = [t[i] for i in f]
                gain = delta_gini_index(left_y, right_y, alpha, branch_count)
                push!(gains, gain)
                if length(rules) == 0
                    rules = [feature_id th]
                end
                rules = vcat(rules, [feature_id th])
            end
        end
        best_rule = rules[argmin(gains), :]
        best_gini = minimum(gains)
        feature_id = Int(best_rule[1])
        threshold = best_rule[2]
        split = x[:, feature_id] .< threshold
        newaxis = [CartesianIndex()]
        left_x, left_t, right_x, right_t = [], [], [], []
        global left_x, left_t, right_x, right_t
        for i in 1 : length(split)
            if split[i]
                if length(left_x) == 0
                    left_x = x[i, :]
                    left_x = left_x[newaxis, :]
                    left_t = t[:, i]
                    left_t = left_t[newaxis, :]
                else
                    kx = x[i, :]
                    kx = kx[newaxis, :]
                    left_x = vcat(left_x, kx)
                    kx = t[:, i]
                    kx = kx[newaxis, :]
                    left_t = vcat(left_t, kx)
                end
            else
                if length(right_x) == 0
                    right_x = x[i, :]
                    right_x = right_x[newaxis, :]
                    right_t = t[:, i]
                    right_t = right_t[newaxis, :]
                else
                    kx = x[i, :]
                    kx = kx[newaxis, :]
                    right_x = vcat(right_x, kx)
                    kx = t[:, i]
                    kx = kx[newaxis, :]
                    right_t = vcat(right_t, kx)
                end
            end
        end
        return left_x, left_t, right_x, right_t, feature_id, threshold, best_gini
    end
    function grow(x, t, classes, gini, alpha, branch_count)
        if size(x)[1] == size(t)[1]
            t = t'
        end
        uniques, counts = unique(t, count = true)
        counter = Dict(zip(uniques, counts))
        class_count = [(c in keys(counter)) ? counter[c] : 0 for c in classes]
        this = Dict("class_count" => class_count, "feature_id" => nothing, "threshold" => nothing, "left" => nothing, "right" => nothing)
        if length(t) == 1
            return this
        elseif length(uniques) == 1
            return this
        end
        a = true
        for i in 1:size(x)[2]
            a = a && length(unique(x[:, i])) == 1
        end
        if a
            return this
        end
        left_x, left_t, right_x, right_t, feature_id, threshold, best_gini = branch(x, t, alpha, branch_count)
        if best_gini < gini
            branch_count += 1
            left = grow(left_x, left_t, classes, best_gini, alpha, branch_count)
            right = grow(right_x, right_t, classes, best_gini, alpha, branch_count)
            return Dict("class_count" => class_count, "feature_id" => feature_id, "threshold" => threshold, "left" => left, "right" => right)
        else
            return this
        end
    end
    classes = unique(t)
    return grow(x, t, classes, Inf, alpha, 1), classes
end
function tree_predict(xs, tree, classes)
    function predict_one(x, tree)
        while tree["feature_id"] != nothing
            tree = (x[tree["feature_id"]] < tree["threshold"]) ? tree["left"] : tree["right"]
        end
        return argmax(tree["class_count"])
    end
    predicts = [predict_one(xs[i, :], tree) for i in 1:size(xs)[1]]
    return predicts
end
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
function fit(x, t; n_trees = 10, alpha = 0.01)
    classes = unique(t)
    forest = Array{Any}(undef, n_trees)
    classes = Array{Any}(undef, n_trees)
    bootstrapped_x, bootstrapped_t, using_feature = bootstrap(x, t, n_trees)
    for i in 1:n_trees
        forest[i], classes[i] = DecisionTree(bootstrapped_x[:, :, i], bootstrapped_t[i, :], alpha = alpha)
    end
    return forest, classes, using_feature
end
function predict(x, forest, classes, using_feature)
    solution = []
    for (tree, classes, feature) in zip(forest, classes, using_feature)
        s = tree_predict(x[:, feature], tree, classes)
        if solution == []
            solution = s
        else    
            solution = hcat(solution, s)
        end
    end
    predicts = []
    for i in 1:length(solution[:, 1])
        class, counts = unique(solution[i, :], count = true)
        push!(predicts, class[argmax(counts)[2]])
    end
    d = zeros(Int, size(classes[1]))
    for i in 1 : length(d)
        d[i] = i
    end
    d = Dict(zip(d, classes[1]))
    for i in 1 : length(predicts)
        predicts[i] = d[predicts[i]]
    end
    return predicts
end