struct Conv{F, W, B, N, M} <: Param
    σ::F
    w::W
    b::B
    stride::NTuple{N, Int}
    padding::NTuple{M, Int}
end

function Conv(k::NTuple{N, Int}, io::Pair{<:Integer, <:Integer}, σ; stride = 1, padding = 0, set_w = "Xavier") where {N}
    weight = conv_w(k, io, set_w)
    n = ndims(weight)
    padding = convert(2N, padding)
    stride = convert(N, stride)
    b = create_bias(weight, size(weight, n))
    return Conv(σ, weight, b, stride, padding)
end

function Base.show(io::IO, C::Conv)
    k = size(C.w)[1:end-2]
    i, o = size(C.w)[end-1:end]
    σ = C.σ
    print("Convolution(k:$k, IO:$i => $o, σ:$σ)")
end

function (C::Conv)(x::AbstractArray)
    M = size(C.w)[end]
    weight = reshape(C.w, div(length(C.w), M), M)
    IC = Im2Col(x, size(C.w)[1:end-2], C.stride, C.padding)
    if size(weight)[1] != size(IC.x)[2]
        kc = C.w[end-1]
        ic = C.c
        throw(DimensionMismatch("the number of channels must match between the kernel and the input!(now kernel: $kc, input: $ic)"))
    end
    σ, b = C.σ, C.b
    x = reshape(σ.(IC.x * weight .+ b'), IC.Oh, IC.Ow, M, IC.b)
    return permutedims(x, [1, 2, 4, 3])
end