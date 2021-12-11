abstract type DimInfo{N, S, P, D} end

inserted_dims(c::DimInfo{N,S,P,D}) where {N, S, P, D} = N
stride(c::DimInfo{N,S,P,D}) where {N, S, P, D} = S
padding(c::DimInfo{N,S,P,D}) where {N, S, P, D} = P
dilation(c::DimInfo{N,S,P,D}) where {N, S, P, D} = D

function output_size(c::ConvDims)
    I = input_size(c)
    K = kernel_size(c)
    S = stride(c)
    P = padding(c)
    D = dilation(c)

    return ntuple(inserted_dims(c)) do i
        return div(I[i] + P[(i-1)*2 + 1] + P[(i-1)*2 + 2] - (K[i] - 1) * D[i] - 1, S[i]) + 1
    end
end

struct ConvDimInfo{N, K, C_in, C_out, G, S, P, D} <: DimInfo{N, S, P, D}
    I::NTuple{N, Int}
end

function ConvDimInfo(c::ConvDims; N=inserted_dims(c), I=input_size(c), K=kernel_size(c),
                       C_in=channels_in(c), C_out=channels_out(c), S=stride(c),
                       P=padding(c), D=dilation(c), G=groupcount(c))
    return ConvDimInfo{N, K, C_in, C_out, G, S, P, D}(I)
end

struct PoolDimInfo{N, S, P, D} <: DimInfo
end

function basetype(::Type{C}) where {C<:DimInfo}
    if C <: ConvDimInfo
        return ConvDimInfo
    else
        return PoolDimInfo
    end
end

input_size(c::ConvDimInfo) = c.I
kernel_size(c::ConvDimInfo{N,K,C_in,C_out,G}) where {N,K,C_in,C_out,G} = K
channels_in(c::ConvDimInfo{N,K,C_in,C_out,G}) where {N,K,C_in,C_out,G} = C_in::Int
channels_out(c::ConvDimInfo{N,K,C_in,C_out,G}) where {N,K,C_in,C_out,G} = C_out::Int
group_count(c::ConvDimInfo{N,K,C_in,C_out,G}) where {N,K,C_in,C_out,G} = G::Int

@inline function insert_dimension(cdims::C) where {D<:DimInfo}
    return basetype(C)(cdims;
        N=inserted_dims(cdims) + 1,
        I=(input_size(cdims)..., 1),
        K=(kernel_size(cdims)..., 1),
        S=(stride(cdims)..., 1),
        P=(padding(cdims)..., 0, 0),
        D=(dilation(cdims)..., 1),
    )
end

@inline function insert_dimension(x::AbstractArray{TX, 3}) where {TX}
    return reshape(x, size(x, 1), 1, size(x, 2), size(x, 3))
end

@inline function insert_dimension(x::AbstractArray{TX, 4}) where {TX}
    return reshape(x, size(x, 1), siize(x, 2), 1, size(x, 3), size(x, 4))
end

@inline function insert_dimension(x, n::Int64)
    for i in n
        x = insert_dimension(x)
    end
    return x
end