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
    recode = m(d[1], rf = true)
    grads = Dict()
    l = length(layers)
    r = recode[l+1]
    Δ = loss(c_t(r[2])..., d[2], back = true)
    σ = layers[l].activation
    Δ = Δ .* σ.(r[1])
    r = recode[l]
    for i in l : -1 : 1
        r = recode[i]
        grads[ps.ps[i*2-1]] = Δ * r[2]'
        grads[ps.ps[i*2]] = Δ
        i == 1 && continue
        Δ = layers[i](Δ, r[1], true)
    end
    Grads(grads, ps.ps)
end