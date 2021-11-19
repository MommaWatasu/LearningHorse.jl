"""
    MaxPool(k::NTuple; stride = k, padding = 0)
This is a layer for max pooling with kernel size `k`.

Expects as input an array with `ndims(x) == N+2`, i.e. channel and batch dimensions, after the `N` feature dimensions, where `N = length(out)`.

The default stride is the same as kernel size `k`.

# Example
```
julia> N = NetWork(Conv((2, 2), 5=>2, relu), MaxPool(2, 2))
Layer1 : Convolution(k:(2, 2), IO:5=>2, σ:relu)
Layer2 : MaxPool(k:(2, 2), stride:(2, 2) padding:(0, 0, 0, 0))

julia> x = rand(Float64, 10, 10, 5, 5) |> size
(10, 10, 5, 5)

julia> N(x) |> size
(4, 4, 5, 2)
```
"""
mutable struct MaxPool{N, M} <: NParam
    k::NTuple{N, Int}
    stride::NTuple{N, Int}
    padding::NTuple{M, Int}
    record::Array
    shape::IOShape
end

function MaxPool(k::NTuple{N, Int}; stride = k, padding = 0) where N
    if N == 1
        k = (k[1], 1)
    end
    stride = convert(2, stride)
    padding = convert(4, padding)
    return MaxPool(k, stride, padding, [], IOShape())
end

function Base.show(io::IO, MP::MaxPool)
    print("MaxPool(k:"*string(MP.k)*", stride:"*string(MP.stride)*", padding:"*string(MP.padding)*")")
end

function (mp::MaxPool)(x::Array{T, N}) where {T, N}
    if N != 4
        x = x[:, :, :, :]
    end
    IC = Im2Col(x, mp.k, mp.stride, mp.padding, trans = "Pool")
    return reshape(maximum(IC.x, dims = 2), IC.Oh, IC.Ow, IC.b, IC.c)
end

function (mp::MaxPool)(x::Array{T, N}, record::Array, i) where {T, N}
    if N != 4
        x = x[:, :, :, :]
    end
    Ih, Iw = size(x)[1:2]
    IC = Im2Col(x, mp.k, mp.stride, mp.padding, trans = "Pool")
    mp.shape(Ih, Iw, IC.Oh, IC.Ow)
    m = argmax(IC.x, dims = 2)
    r = zero(IC.x)
    for i in m r[i] = 1 end
    mp.record = r
    m = reshape(IC.x[m], IC.Oh, IC.Ow, IC.b, IC.c)
    record[i+1] = (m, m)
    return m
end

function (mp::MaxPool)(Δ, z, back::Bool)
    k = mp.k
    x = reshape(permutedims(Δ, [1, 3, 4, 2]), length(Δ), 1)
    C, B = size(z)[3:4]
    Ih, Iw = mp.shape.IShape
    Oh, Ow = mp.shape.OShape
    Δ = reshape(x .* mp.record, B*Oh*Ow, C*k[1]*k[2])' 
    return Col2Im(Δ, k, mp.shape, mp.stride, mp.padding).x
end

trainable(M::MaxPool) = tuple()