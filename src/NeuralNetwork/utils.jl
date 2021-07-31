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

mutable struct IOShape
    IShape::Tuple
    OShape::Tuple
    function IOShape()
        new(tuple(), tuple())
    end
end

function (IOS::IOShape)(Ih, Iw, Oh, Ow)
    IOS.IShape = (Ih, Iw)
    IOS.OShape = (Oh, Ow)
end

struct Params
    ps::Dict
    l::Int
    function Params(model)
        ps = Dict()
        layers = model.net
        for i in 1 : length(layers)
            l = length(ps)
            if typeof(layers[i]) <: Param
                ps[l+1], ps[l+2] = layers[i].w, layers[i].b
            else
                ps[l+1], ps[l+2] = nothing, nothing
            end
        end
        new(ps, length(ps))
    end
end

function Base.iterate(P::Params, state = 1)
    if state > P.l
        return nothing
    else
        (P.ps[P.l-state+1], state + 1)
    end
end