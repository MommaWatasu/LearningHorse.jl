function line(x)
    return x
end

line(x, back::Bool) = 1

function sigmoid(x)
    t = exp(-abs(x))
    return (x>=0) ? inv(1 + t) : t / (t + 1)
end

sigmoid(x, back::Bool) = sigmoid(x)*(1-sigmoid(x))

hardsigmoid(x) = max(0, min(1, (x + 2.5) / 6))

function hardsigmoid(x, back::Bool)
    if -2.5 <= x <= 2.5
        return 0.2
    else
        return 0
    end
end

relu(x) = max(0, x)

function relu(x, back::Bool)
    if x <= 0
        return 0
    else
        return 1
    end
end

function elu(x; alpha = 1)
    return (x > 0) ? x : (exp(x) - 1)alpha
end

function elu(x, back::Bool; alpha = 1)
    if x < 0
        return alpha*exp(x)
    else
        return 1
    end
end

function selu(x)
    λ = oftype(float(x), selu_λ)
    α = oftype(float(x), selu_α)
    return ((x > 0) ? x : (exp(x)-1)α)*λ
end

function selu(x, back::Bool)
    λ = oftype(float(x), selu_λ)
    α = oftype(float(x), selu_α)
    if x < 0
        return λ*alpha*exp(x)
    else
        return λ
    end
end

const selu_λ = 1.0507009873554804934193349852946
const selu_α = 1.6732632423543772848170429916717

function tanh(x)
    return exp(x) - exp(-x) / exp(x) + exp(-x)
end

tanh(x, back::Bool) = 4 / (exp(x) + exp(-x)) ^ 2

function hardtanh(x)
    if x > 1
        return 1
    elseif x < -1
        return -1
    else
        return x
    end
end

function hardtanh(x, back::Bool)
    if x < -1 || 1 <= x
        return 0
    else
        return 1
    end
end

function softsign(x)
    return x / (1 + abs(x))
end

softsign(x, back::Bool) = 1 / ((1 + abs(x))^2)

function softplus(x)
    return log(1 + exp(x))
end

softplus(x, back::Bool) = 1 / (1 + exp(-x))