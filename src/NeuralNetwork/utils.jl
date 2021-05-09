function dense_w(in_size, out_size, m)
    if m == "Xavier"
        return randn((out_size, in_size)) ./ sqrt(in_size)
    elseif m == "He"
        return randn((out_size, in_size)) ./ sqrt(in_size) .* sqrt(2)
    else
        throw(ArgumentError("only'Xivier'or'He' can be specified as the initial value setting method."))
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
        throw(ArgumentError("i is Int type, Tuple or 'same'"))
    end
    return i
end

struct Params
    ps::Dict
    function Params(model)
        ps = Dict()
        layers = model.net
        for i in 1 : length(layers)
            if typeof(layers[i]) <: Param
                weight, bias = layers[i].w, layers[i].b
                l = length(ps)
                ps[l+1], ps[l+2] = weight, bias
            end
        end
        new(ps)
    end
end

function Base.iterate(P::Params, state = 1)
    if state > length(P.ps)
        return nothing
    else
        (P.ps[state], state + 1)
    end
end