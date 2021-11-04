struct Grads
    g::Array{AbstractVecOrMat{Float64}}
    ps::Array
end

function Base.getindex(G::Grads, t::T) where T <:AbstractArray
    return G.g[t]
end

c_t(x::Tuple) = x
c_t(x) = Tuple(x)

function grad(m::NetWork, d, loss, ps::Params)
    layers = m.net
    record::Array{Tuple, 1} = m(d[1], rf = true)
    l = length(layers)
    j = l*2
    grads = Array{AbstractVecOrMat{Float64}}(undef, l*2)
    r::Tuple{Array, Array} = record[l+1]
    Δ::AbstractVecOrMat = loss(r[2], d[2], true)
    σ = layers[l].σ
    @. Δ = Δ * σ(r[1], true)
    for i in l : -1 : 1
        r = record[i]
        if typeof(layers[i]) <: Conv
            C::Conv = layers[i]
            Δ = c_dims(layers[i], Δ)
            _r::Array{Float64, 2} = Im2Col(r[2], size(C.W)[1:end-2], C.stride, C.padding).x
            grads[j-1] = Δ * _r'
            grads[j] = dropdims(sum(Δ, dims = 2), dims = 2)
            j-=2
        elseif typeof(layers[i]) <: Param
            grads[j-1] = Δ * r[2]'
            grads[j] = Δ
            j-=2
        end
        i == 1 && continue
        Δ = layers[i](Δ, r[1], true)
    end
    Grads(grads, ps.ps)
end

function c_dims(C::Conv, x)
    oc = size(C.W)[end]
    other = Int(length(x) / oc)
    return reshape(x, oc, other)
end