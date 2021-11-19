"""
    SVC(; alpha=0.01, ni=1000)
Support Vector Machine Classifier.

This struct learns classifiers using One-Vs-Rest.
One-Vs-Rest generates two-class classifiers divided into one class and the other classes using Logistic Regression, adopting the most likely one among all classifiers.

Parameter `α` indicates the learning rate, and `ni` indicates the number of learnings.
"""
mutable struct SVC
    α::Float64
    n_iter::Int64
    classifiers::Array{Logistic, 1}
    SVC(; alpha=0.01, ni=1000) = new(alpha, ni, Array{Logistic}(undef, 0))
end

function fit!(model::SVC, x, t)
    alpha, n_iter = model.α, model.n_iter
    check_size(x, t)
    c = size(t, 2)
    w = Array{Float64}(undef, 0, 0)
    classifiers = Array{Logistic}(undef, 0)
    for i in 1 : c
        classifier = Logistic(alpha = alpha, ni = n_iter)
        OHE = OneHotEncoder()
        fit!(classifier, x, OHE(t[:, i]))
        push!(classifiers, classifier)
    end
    model.classifiers = classifiers
end

function predict(model::SVC, x)
    p = Array{Float64}(undef, 0, 0)
    for i in 1 : length(model.classifiers)
        if i == 1
            p = forecast(model.classifiers[i], x)[:, 2]
        else
            p = hcat(p, forecast(model.classifiers[i], x)[:, 2])
        end
    end
    return [findfirst(p[i, :] .== maximum(p[i, :])) for i in 1:size(p, 1)]
end
