struct Grads
    g::Dict
    ps::Dict
end

c_t(x::Tuple) = x
c_t(x) = Tuple(x)

function grad(m, d, loss, ps)
    recode = m(d, rf = true)
    grads = copy(ps.ps)
    Δ = fill!(similar(recode[end]), loss)
    for i in length(N.net) : 1
        z = recode[i]
        grads[i] = (z*Δ, Δ)
        i == 1 || continue
        Δ = layers[i](x, z, back = true)
    end
    Grads(grads, ps.ps)
end