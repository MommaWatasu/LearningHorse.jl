function conv_im2col!(
        y::AbstractArray{T, 5}, x::AbstractArray{T, 5}, w::AbstractArray{T, 5},
        diminfo::ConvDimInfo; col::AbstractArray{T, 3}=similar(x, im2col_dims(diminfo)),
        alpha::T=T(1), beta::T=T(0)) where {T}
    @threads for batch_index in 1 : size(x, 5)
    end
end