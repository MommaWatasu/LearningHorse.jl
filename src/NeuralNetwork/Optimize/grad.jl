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
    Δ = fill!(similar(recode[end]), loss)
    for i in length(m.net) : -1 : 1
        z = recode[i]
        grads[ps.ps[i*2-1]] = Δ * z'
        grads[ps.ps[i*2]] = Δ
        i == 1 && continue
        Δ = layers[i](Δ, z, true)
    end
    Grads(grads, ps.ps)
end