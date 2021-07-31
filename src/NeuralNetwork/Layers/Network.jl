using Random
abstract type Layer end
#This type isn't Optimized.
abstract type NParam <: Layer end
#This type is Optimized.
abstract type Param <: Layer end

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
        record = Dict()
        layers = N.net
        record[1] = (nothing, x)
        for i in 1 : length(layers)
            x = layers[i](x, record, i)
        end
        return record
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
    w::AbstractArray
    b::AbstractArray
    record::AbstractArray
    activation::F
end

function Dense(io::Pair{<:Integer, <:Integer}, activation; set_w = "Xavier", set_b = zeros)
    return Dense(dense_w(io..., set_w), set_b(io[2]), [], activation)
end

function Base.show(io::IO, D::Dense)
    o, i = size(D.w)
    σ = D.activation
    print("Dense(IO:$i => $o, σ:$σ)")
end

function (layer::Dense)(x::AbstractArray)
    w, b, σ = layer.w, layer.b, layer.activation
    σ.(w * x + b)
end

function (layer::Dense)(x::AbstractArray, record::Dict, i)
    w, b, σ = layer.w, layer.b, layer.activation
    z = w * x
    a = σ.(z + b)
    record[i+1] = (z, a)
    a
end

function (layer::Dense)(Δ, z, back::Bool)
    w, σ = layer.w, layer.activation
    Σ = zeros(size(w)[2])
    for i in 1 : length(Σ)
        Σ[i] += w[:, i] ⋅ Δ
    end
    Σ.*σ.(z, true)
end

mutable struct Flatten <: NParam
    csize::Tuple
    function Flatten()
        return new(())
    end
end

function (F::Flatten)(x)
    F.csize = size(x)
    return reshape(x, length(x))
end

function (F::Flatten)(x, record::Dict, i)
    F.csize = size(x)
    a = reshape(x, length(x))
    record[i+1] = (a, a)
    return a
end

(F::Flatten)(Δ, z, back::Bool) = reshape(Δ, F.csize...)

struct Dropout <: NParam
    p::Float64
end

function Base.show(io::IO, D::Dropout)
    print("Dropout(")
    print(string(D.p))
    print(")")
end

dropout_kernel(y::T, p, q) where {T} = (y > p) ? float(1 / q) : float(0)

function (D::Dropout)(x)
    y = rand!(similar(x))
    y = dropout_kernel.(y, D.p, 1 - D.p)
    x .* y
end

function (D::Dropout)(x::AbstractArray, record::Dict, i)
    a = D(x)
    record[i+1] = (a, a)
    a
end

(D::Dropout)(Δ, z, back::Bool) = Δ

function add_layer!(model::NetWork, obj::T) where T<: Layer
    l = length(model.net)
    model.net[l+1] = obj
end