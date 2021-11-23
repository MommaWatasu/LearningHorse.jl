function dense_w(in_size, out_size, m)
    if m == "Xavier"
        return randn((out_size, in_size)) ./ sqrt(in_size)
    elseif m == "He"
        return randn((out_size, in_size)) ./ sqrt(in_size) .* sqrt(2)
    else
        try
            return m(out_size, in_size)
        catch
            throw(ArgumentError("`set_w` must be `Xavier`, `He` or a function to create weight."))
        end
    end
end

function conv_w(K, io, m)
    if m == "Xavier"
        return randn(K..., io...) ./ sqrt(io[1])
    elseif m == "He"
        return randn(K..., io...) ./ sqrt(io[1]) .* sqrt(2)
    else
        throw(ArgumentError("only'Xivier'or'He' can be specified as the initial value setting method."))
    end
end

create_bias(w, dims...) = fill!(similar(w, dims...), 0)

convert(N, i::Integer) = ntuple(_->i, N)
convert(N, i::Tuple) = i
function convert(N, i::String)
    if i != "same"
        throw(ArgumentError("`i` is Int type, Tuple or `same`"))
    end
    return i
end

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

struct KeepSize end

expand_padding(padding, k::NTuple{N, T}, dilation, stride) where {N, T} = expand(2N, padding)
function expand_padding(::KeepSize, k::NTuple{N, T}, dilation, stride) where {N, T}
    k_eff = @. k + (k - 1) * (dilation - 1)
    pad_amt = @. k_eff - 1
    return Tuple(mapfoldl(i -> [cld(i, 2), fld(i,2)], vcat, pad_amt))
end

function params(model)
    pa = Array{Any}(undef, 0)
    for i in 1 : length(model.net)
        try
            layer = model[i]
            push!(pa, trainable(layer)...)
        catch
            continue
        end
    end
    return Zygote.Params(pa)
end

function check_size(w, x, kc, ic)
    if w != x
        throw(DimensionMismatch("the number of channels must match between the kernel and the input!(now kernel: $kc, input: $ic)"))
    end
end

function dim_check(p::Tuple, name::String)
    if length(p) != nd
        throw(DimensionMismatch("$(name) $(length(p))d, shoud be $(nd)d!"))
    end
end