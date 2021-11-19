export line, sigmoid, hardsigmoid, relu6, relu, elu, selu, hardtanh, softsign, softplus, mish, swish

"""
    line(x) = x
Very simple activation function. 
"""
function line(x)
    return x
end

line(x, back::Bool) = 1

"""
    sigmoid(x) = 1 / (1 + exp(-x))
Standard sigmoid activation function.
"""
function sigmoid(x)
    t = exp(-abs(x))
    (x>=0) ? inv(1 + t) : t / (t + 1)
end

sigmoid(x, back::Bool) = sigmoid(x)*(1-sigmoid(x))

"""
    hardsigmoid(x) = max(0, min(1, (x + 2.5) / 6))
Piecewise linear approximation of sigmoid.
"""
hardsigmoid(x) = max(0, min(1, (x + 2.5) / 6))

function hardsigmoid(x, back::Bool)
    if -2.5 <= x <= 2.5
        return 0.2
    else
        return 0
    end
end

"""
    relu(x) = max(0, x)
Standard relu activation function.
"""
relu(x) = max(0, x)

function relu(x, back::Bool)
    if x <= 0
        return 0
    else
        return 1
    end
end

"""
    relu6(x) = (x>=6) ? 6 : max(0, x)
Relu function with an upper limit of 6.
"""
relu6(x) = (x>=6) ? 6 : max(0, x)

relu6(x, back::Bool) = (x<=0||x>=6) ? 0 : 1

"""
    elu(x, alpha = 1) = 
Exponential Linear Unit activation function. You can also specify the coefficient explicitly, e.g. elu(x, 1).
"""
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

"""
    selu(x) = λ * (x ≥ 0 ? x : α * (exp(x) - 1))
    
    λ = 1.0507009873554804934193349852946
    α = 1.6732632423543772848170429916717
Scaled exponential linear units
"""
function selu(x)
    λ = oftype(float(x), selu_λ)
    α = oftype(float(x), selu_α)
    return ((x > 0) ? x : (exp(x)-1)α)*λ
end

function selu(x, back::Bool)
    λ = oftype(float(x), selu_λ)
    α = oftype(float(x), selu_α)
    if x < 0
        return λ*α*exp(x)
    else
        return λ
    end
end

const selu_λ = 1.0507009873554804934193349852946
const selu_α = 1.6732632423543772848170429916717

"""
    tanh(x) = exp(x) - exp(-x) / exp(x) + exp(-x)
Hyboic tanhgent funciton.
"""
Base.tanh(x, back::Bool) = 4 / (exp(x) + exp(-x)) ^ 2

"""
    hardtanh(x) = (x>=1) ? 1 : max(-1, x)
Linear tanh function.
"""
hardtanh(x) = (x>=1) ? 1 : max(-1, x)

function hardtanh(x, back::Bool)
    if x < -1 || 1 <= x
        return 0
    else
        return 1
    end
end

"""
    softsign(x) = x / (1+abs(x))
The softplus activation function.
"""
function softsign(x)
    return x / (1 + abs(x))
end

softsign(x, back::Bool) = 1 / ((1 + abs(x))^2)

"""
    softplus(x) = log(1 + exp(x))
the softplus activation function.
"""
function softplus(x)
    return log(1 + exp(x))
end

softplus(x, back::Bool) = 1 / (1 + exp(-x))

"""
    mish(x) = x * tanh(softplus(x))
The mish function.
"""
mish(x) = x * tanh(softplus(x))

omega(x) = 4*(x+1)+4*exp(2x)+exp(3x)+exp(x)*(4x+6)

delta(x) = 2*exp(x)+exp(2x)+2

mish(x, back::Bool) = exp(x)*omega(x)/delta(x)^2

"""
    swish(x) = x * sigmoid(x)
The swish function.
"""
swish(x) = x * sigmoid(x)

swish(x, back::Bool) = swish(x) + (sigmoid(x) * (1-swish(x)))