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

function (mp::MaxPool)(x::Array{T, N}, record::Dict, i) where {T, N}
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
    record[i+1] = (x, m)
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