struct Grads
    g::Dict
    ps::Dict
end

function Base.getindex(G::Grads, t::T) where T <:AbstractArray
    return G.g[t]
end

c_t(x::Tuple) = x
c_t(x) = Tuple(x)

function grad(m, d, loss, ps)
    layers = m.net
    record = m(d[1], rf = true)
    grads = Dict()
    l = length(layers)
    r = record[l+1]
    Δ = loss(r[2], d[2], back = true)
    σ = layers[l].activation
    Δ = Δ .* σ.(r[1])
    for i in l : -1 : 1
        r = record[i]
        if typeof(layers[i]) <: Conv
            C = layers[i]
            Δ = c_dims(layers[i], Δ)
            _r = Im2Col(r[2], size(C.W)[1:end-2], C.stride, C.padding).x
            grads[ps.ps[i*2-1]] = Δ * _r'
            grads[ps.ps[i*2]] = dropdims(sum(Δ, dims = 2), dims = 2)
        elseif typeof(layers[i]) <: Param
            grads[ps.ps[i*2-1]] = Δ * r[2]'
            grads[ps.ps[i*2]] = Δ
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