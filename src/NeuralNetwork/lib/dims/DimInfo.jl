abstract type DimInfo{N, S, P, D} end

spatial_dims(c::DimInfo{N,S,P,D}) where {N, S, P, D} = N
stride(c::DimInfo{N,S,P,D}) where {N, S, P, D} = S
padding(c::DimInfo{N,S,P,D}) where {N, S, P, D} = P
dilation(c::DimInfo{N,S,P,D}) where {N, S, P, D} = D

function output_size(c::ConvDims)
    I = input_size(c)
    K = kernel_size(c)
    S = stride(c)
    P = padding(c)
    D = dilation(c)

    return ntuple(spatial_dims(c)) do i
        return div(I[i] + P[(i-1)*2 + 1] + P[(i-1)*2 + 2] - (K[i] - 1) * D[i] - 1, S[i]) + 1
    end
end

struct ConvDimInfo{N, K, C_in, C_out, S, P, D}
    I::NTuple{N, Int}
end

input_size(c::ConvDimInfo) = c.I
kernel_size(c::ConvDimInfo{N,K,C_in,C_out}) where {N,K,C_in,C_out} = K
channels_in(c::ConvDimInfo{N,K,C_in,C_out}) where {N,K,C_in,C_out} = C_in::Int
channels_out(c::ConvDimInfo{N,K,C_in,C_out,G}) where {N,K,C_in,C_out,G} = C_out::Int

@inline function insert_dimension(cdims::C) where {C <: ConvDims}
    return basetype(C)(cdims;
        N=spatial_dims(cdims) + 1,
        I=(input_size(cdims)..., 1),
        K=(kernel_size(cdims)..., 1),
        S=(stride(cdims)..., 1),
        # Padding is always the problem child....
        P=(padding(cdims)..., 0, 0),
        D=(dilation(cdims)..., 1),
    )
end

@inline function insert_dimension(x::AbstractArray{TX, 3})
    
end