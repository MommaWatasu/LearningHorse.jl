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
    ps::Array{Union{Nothing, AbstractVecOrMat{Float64}}, 1}
    function Params(model)
        layers = model.net
        l = length(layers)
        j = 1
        ps = Array{Union{Nothing, AbstractVecOrMat{Float64}}}(undef, l*2)
        for i in 1 : length(layers)
            L = layers[i]
            if typeof(L) <: Param
                ps[j] = L.w
                ps[j+1] = L.b
                j+=2
            else
                ps[j] = nothing
                ps[j+1] = nothing
                j+=2
            end
        end
        new(ps)
    end
end

function Base.iterate(P::Params, state = 1)
    if state > P.l
        return nothing
    else
        (P.ps[P.l-state+1], state + 1)
    end
end

function zygote_params(model)
    pa = Array{Any}(undef, 0)
    for i in 1 : length(model.net)
        layer = model.net[i]
        push!(pa, trainable(layer)...)
    end
    return Zygote.Params(pa)
end