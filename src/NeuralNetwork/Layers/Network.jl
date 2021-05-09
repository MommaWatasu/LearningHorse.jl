using Random
abstract type Layer end
# Param is the type which has field 'weight' and 'bias'
abstract type Param  <: Layer end
# NParam is the type which has field 'weight' and 'bias'
abstract type NParam <: Layer end

include("../utils.jl")
include("../activations.jl")
include("./Pooling.jl")
include("./Conv.jl")

struct NetWork
    net::Dict
    function NetWork()
        new(Dict())
    end
end

function (N::NetWork)(x; rf=false)
    if !rf
        layers = N.net
        for i in 1 : length(layers)
            x = layers[i](x)
        end
        return x
    else
        recode = []
        layers = N.net
        push!(recode, x)
        for i in 1 : length(layers)
            x = layers[i](x)
            push!(recode, x)
        end
        return recode
    end
end

function Base.show(io::IO, N::NetWork)
    layers = N.net
    for i in 1 : length(layers)
        print("Layer$i : ")
        if i != length(layers)
            println(layers[i])
        else
            print(layers[i])
        end
    end
end

struct Dense{F} <: Param
    w::T where T <: AbstractArray
    b::T where T <: AbstractArray
    activation::F
end

function Dense(io::Pair{<:Integer, <:Integer}, activation; set_w = "Xavier", set_b = zeros)
    return Dense(dense_w(io..., set_w), set_b(io[2]), activation)
end

function Base.show(io::IO, D::Dense)
    o, i = size(D.w)
    σ = D.activation
    print("Dense(IO:$i => $o, σ:$σ)")
end

function (layer::Dense)(x::T) where T<:AbstractArray
    w, b, σ = layer.w, layer.b, layer.activation
    σ.(w * x + b)
end

function (layer::Dense)(Δ, z, back::Bool)
    w, σ = layer.w, layer.activation
    Σ = zeros(size(w)[2])
    for i in 1 : length(Σ)
        Σ[i] += w[:, i] ⋅ Δ
    end
    Σ.*σ.(z, true)
end

struct Flatten <: NParam end

function (F::Flatten)(x)
    return reshape(x, length(x))
end

mutable struct Dropout <: NParam
    p::Float64
    active::Bool
end

function Base.show(io::IO, D::Dropout)
    print("Dropout(")
    print(string(D.p))
    print(")")
end

function Dropout(p; active = false)
    return Dropout(p, active)
end

dropout_kernel(y::T, p, q) where {T} = (y > p) ? float(1 / q) : float(0)

function (D::Dropout)(x)
    D.active || return x
    y = rand!(similar(x))
    y = dropout_kernel.(y, D.p, 1 - D.p)
    return x .* y
end

function add_layer!(model::NetWork, obj::T) where T<: Layer
    l = length(model.net)
    model.net[l+1] = obj
end