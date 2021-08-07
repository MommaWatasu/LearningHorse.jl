"""
    Conv(kernel, in=>out, σ; stride = 1, pading = 0, set_w = "Xavier")
This is the traditional convolution layer. `kernel` is a tuple of integers that specifies the kernel size, it must have one or two elements. And, `in` and `out` specifies number of input and out channels.

The input data must have a dimensions WHCB(weight, width, channel, batch). If you want to use a data which has a dimentions WHC, you must be add a dimentioins of B.

`stride` and `padding` are single integers or tuple(stride is tuple of 2 elements, padding is tuple of 2 elements), and if you specifies `same` to padding, we adjust sizes of input and return a matrix which has the same sizes.
`set_w` is `Xavier` or `He`, it decide a method to create a first parameter. This parameter is the same as `Dense()`.

# Example
```
julia> C = Conv((2, 2), 2=>2, relu)
Convolution(k:(2, 2), IO:2 => 2, σ:relu)

julia> C(rand(10, 10, 2, 5)) |> size
(9, 9, 2, 5)
```

!!! warning
    When you specidies `same` to `padding`, in some cases, it will be returned one size smaller.
    Because of its expression.
```

julia> C = Conv((2, 2), 2=>2, relu, padding = "same")
Convolution(k:(2, 2), IO:2 => 2, σ:relu

julia> C(rand(10, 10, 2, 5)) |> size
(9, 9, 2, 5)
```
"""
struct Conv <: Param
    σ
    w
    W
    b
    stride::Tuple
    padding::Union{Tuple, String}
    shape::IOShape
end

function Conv(k::NTuple{N, Int}, io::Pair{<:Integer, <:Integer}, σ; stride = 1, padding = 0, set_w = "Xavier") where {N}
    weight = conv_w(k, io, set_w)
    M = io[2]
    w = reshape(weight, M, div(length(weight), M))
    n = ndims(weight)
    padding = convert(2N, padding)
    stride = convert(N, stride)
    b = create_bias(weight, size(weight, n))
    return Conv(σ, w, weight, b, stride, padding, IOShape())
end

function Base.show(io::IO, C::Conv)
    k = size(C.W)[1:end-2]
    i, o = size(C.W)[end-1:end]
    σ = C.σ
    print("Convolution(k:$k, IO:$i => $o, σ:$σ)")
end

function (C::Conv)(x::AbstractArray)
    IC = Im2Col(x, size(C.W)[1:2], C.stride, C.padding)
    if size(C.w)[2] != size(IC.x)[1]
        kc = size(C.w)[end-1]
        ic = IC.c
        throw(DimensionMismatch("the number of channels must match between the kernel and the input!(now kernel: $kc, input: $ic)"))
    end
    σ, b = C.σ, C.b
    M = size(C.W)[end]
    x = reshape(σ.(C.w * IC.x .+ b), IC.Oh, IC.Ow, M, IC.b)
    return x
end

function (C::Conv)(x::AbstractArray, record::Dict, i)
    Ih, Iw = size(x)[1:2]
    IC = Im2Col(x, size(C.W)[1:2], C.stride, C.padding)
    C.shape(Ih, Iw, IC.Oh, IC.Ow)
    if size(C.w)[2] != size(IC.x)[1]
        kc = size(C.w)[end-1]
        ic = IC.c
        throw(DimensionMismatch("the number of channels must match between the kernel and the input!(now kernel: $kc, input: $ic)"))
    end
    σ, b = C.σ, C.b
    M = size(C.W)[end]
    z = C.w * IC.x
    _z = reshape(z, IC.Oh, IC.Ow, M, IC.b)
    a = σ.(z .+ b)
    a = reshape(a, IC.Oh, IC.Ow, M, IC.b)
    record[i+1] = (_z, a)
    return a
end

function (C::Conv)(Δ, z, back::Bool)
    z = Im2Col(z, size(C.W)[1:2], C.stride, C.padding).x
    w, σ = C.w, C.σ
    Δ = w' * Δ .* σ.(z, true)
    Fh, Fw,  = size(C.W)
    Δ = Col2Im(Δ, size(C.W)[1:2], C.shape, C.stride, C.padding).x
    #I have to use Col2Im to Δ.
    return Δ
end