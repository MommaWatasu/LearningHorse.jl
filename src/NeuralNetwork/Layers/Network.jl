using Random
abstract type Layer end
#This type isn't Optimized.
abstract type NParam <: Layer end
#This type is Optimized.
abstract type Param <: Layer end
export Param, NParam

include("../activations.jl")
include("./Pooling.jl")
include("./Conv.jl")

"""
    NetWork(layers...)

Connect multiple layers, and build a NeuralNetwork.
NetWork also supports index.
You can also add layers later using the add_layer!() Function.

# Example
```
julia> N = NetWork(Dense(10=>5, relu), Dense(5=>1, relu))

julia> N[1]

Dense(IO:10=>5, σ:relu)
```
"""
struct NetWork
    net::Dict{Int, Layer}
    function NetWork()
        new(Dict{Int, Layer}())
    end
end

function NetWork(layers...)
    N = NetWork()
    add_layer!(N, layers...)
    return N
end

function Base.getindex(N::NetWork, i::Int)
    return N.net[i]
end

function (N::NetWork)(x; rf=false)
    layers = N.net
    if !rf
        for i in 1 : length(layers)
            x = layers[i](x)
        end
        return x
    else
        record = Array{Tuple}(undef, length(layers)+1)
        record[1] = ([0.0], x)
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

"""
    Dense(in=>out, σ; set_w = "Xavier", set_b = zeros)
Crate a traditinal `Dense` layer, whose forward propagation is given by:
    y = σ.(W * x .+ b)
The input of `x` should be a Vactor of length `in`, (Sorry for you can't learn using batch. I'll implement)

# Example
```
julia> D = Dense(5=>2, relu)
Dense(IO:5=>2, σ:relu)

julia> D(rand(Float64, 5)) |> size
(2,)
```
"""
struct Dense{W, B, F} <: Param
    w::W
    b::B
    σ::F
end

function Dense(io::Pair{<:Integer, <:Integer}, σ; set_w = "Xavier", set_b = zeros)
    w, b = dense_w(io..., set_w), set_b(io[2])
    Dense(w, b, σ)
end

trainable(D::Dense) = D.w, D.b

function Base.show(io::IO, D::Dense)
    o, i = size(D.w)
    σ = D.σ
    print("Dense(IO:$i => $o, σ:$σ)")
end

function (D::Dense)(x::AbstractVecOrMat)
    w, b, σ = D.w, D.b, D.σ
    σ.(w*x.+b)
end

function (D::Dense)(x::Array, record::Array, i::Int)
    w, b, σ = D.w, D.b, D.σ
    z::Array = w * x
    a::Array = σ.(z .+ b)
    record[i+1] = (z, a)
    a
end

function (D::Dense)(Δ::Array, z::Array, back::Bool)
    #TODO:Corresponds to a batch of x.
    w, σ = D.w, D.σ
    w'*Δ.*σ.(z, true)
end

"""
    Flatten()
This layer change the dimentions Image to Vector.

# Example
```
julia> F = Flatten()
Flatten(())

julia> F(rand(10, 10, 2, 5)) |> size
(1000, )
```
"""
mutable struct Flatten <: NParam
    csize::Tuple
    function Flatten()
        return new(())
    end
end

function (F::Flatten)(x::Array)
    F.csize = size(x)
    return reshape(x, length(x))
end

function (F::Flatten)(x::Array, record::Array, i)
    F.csize = size(x)
    a = reshape(x, :)
    record[i+1] = (a, a)
    return a
end

trainable(F::Flatten) = tuple()

(F::Flatten)(Δ::Array, z::Array, back::Bool) = reshape(Δ, F.csize...)

"""
    Dropout(p)
This layer dropout the input data.

# Example
julia> D = Dropout(0.25)
Dropout(0.25)

julia> D(rand(10))
10-element Array{Float64,1}:
 0.0
 0.3955865029078952
 0.8157710047424143
 1.0129613533211907
 0.8060508293474877
 1.1067504108970596
 0.1461289547292684
 0.0
 0.04581776023870532
 1.2794087133638332
"""
struct Dropout <: NParam
    p::Float64
end

function Base.show(io::IO, D::Dropout)
    print("Dropout(")
    print(string(D.p))
    print(")")
end

dropout_kernel(y::T, p::Float64, q::Float64) where {T} = (y > p) ? float(1 / q) : float(0)

function (D::Dropout)(x::Array)
    y = rand!(similar(x))
    y = dropout_kernel.(y, D.p, 1 - D.p)
    x .* y
end

function (D::Dropout)(x::AbstractArray, record::Array, i::Int)
    a = D(x)
    record[i+1] = (a, a)
    a
end

(D::Dropout)(Δ::Array, z::Array, back::Bool) = Δ

trainable(D::Dropout) = tuple()

"""
     add_layer!(model, layers...)
This function add layers to model. You can add layers when you create a NeuralNetwork with `NetWork()`, and you can also use this function to add layers later.

# Example
julia> N = NetWork()

julia> add_layer!(N, Dense(10=>5, relu), Dense(5=>1, relu))

julia> N
Layer1 : Dense(IO:10 => 5, σ:relu)
Layer2 : Dense(IO:5 => 1, σ:relu)
"""
function add_layer!(model::NetWork, obj::T) where T <: Layer
    l = length(model.net)
    model.net[l+1] = obj
end

function add_layer!(model::NetWork, layers...)
    for layer in layers
        add_layer!(model, layer)
    end
end

"""
    @epochs n ex
This macro cruns `ex` `n` times. Basically this is useful for learning NeuralNetwork.

# Example
julia> a = 1

julia> @epochs 1000 a+=1
progress:1000/1000
julia>a
1001
"""
macro epochs(n, ex)
    for i in 1 : n
        progress = "progress:"*string(i)*"/"*string(n)*"\r"
        print(progress)
        #@info "Epoch $i"
        eval(ex)
    end
end