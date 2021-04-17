struct MaxPool{N, M} <: NParam
    k::NTuple{N, Int}
    stride::NTuple{N, Int}
    padding::NTuple{M, Int}
end

function MaxPool(k::NTuple{N, Int}; stride = k, padding = 0) where N
    if N == 1
        k = (k[1], 1)
    end
    stride = convert(2, stride)
    padding = convert(4, padding)
    return MaxPool(k, stride, padding)
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