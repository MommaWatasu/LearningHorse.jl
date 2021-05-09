export Descent, Momentum, AdaGrad, Adam

const ε = 1e-8

struct Descent
    eta::Float64
end

Descent() = Descent(0.1)

apply!(opt::Descent, x, Δ) = opt.eta .* Δ

struct Momentum
    eta::Float64
    alpha::Float64
    velocity::Dict
end

Momentum(η = 0.01, α = 0.9) = Momentum(η, α, Dict())

function apply!(opt::Momentum, x, Δ)
    η, α = opt.eta, opt.alpha
    v = get!(() -> zeros(x), opt.velocity, x)::typeof(x) #get! () function returns a view.
    v = (α*v - η*Δ)
    Δ = -v
end

struct AdaGrad
    eta::Dict
    h::Float64
end

AdaGrad(eta = 0.01) = AdaGrad(eta, Dict())

function apply!(opt::AdaGrad, x, Δ)
    η = opt.eta
    h = get!(() -> zeros(x), opt.h, x)::typeof(x)
    @. h += Δ^2
    @. Δ *= η / (√h + ε)
end

struct Adam
    eta::Float64
    beta::Tuple{Float64, Float64}
    recode::Dict
end

Adam(η = 0.01, β = (0.9, 0.99)) = Adam(η, β, Dict())

function aplly!(opt::Adam, x, Δ)
    η, β = opt.eta, opt.beta
    mt, vt , βp = get!(opt.recode, x) do
        (zero(x), zero(x), Float64[β[1], β[2]])
    end
    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ^2
    @. Δ = η * mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ε)
    βp .= βp .* β
    return Δ
end