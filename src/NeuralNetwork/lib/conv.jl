function conv(x::AbstractArray{TX, N}, w::AbstractArray{TW, N}, diminfo::DimInfo) where {TX, TW, N}
    y = similar(x, promote_type(TX, TW), output_size(diminfo)..., channels_out(diminfo), size(x, N))
    conv!(
        insert_dimension(y, 5-N),
        insert_dimension(x, 5-N),
        insert_dimension(w, 5-N),
        insert_dimension(diminfo, 5-N)
    )
    return y
end

function conv!(y::AbstractArray{T, 5}, x::AbstractArray{T, 5}, w::AbstratArray{T, 5}, diminfo::C) where {C<:DimInfo}
    xcs = Iterators.partition(1:size(x, 4),
        div(channels_in(diminfo), group_count(diminfo))) # x_channel_size
    wcs = Iterators.partition(1:size(w, 5),
        div(channels_out(diminfo), group_count(diminfo))) # w_channel_size
    
    new_diminfo = basetype(C)(
        diminfo,
        G = 1,
        C_in = div(channels_in(diminfo), group_count(diminfo)),
        C_out = div(channels_out(diminfo), group_count(diminfo))
    )
    
    for (xc, wc) in zip(xcs, wcs)
        vx = @view x[ntuple(i -> i==4 ? xc : Colon(), 5)...]
        vy = @view y[ntuple(i -> i==5 ? wc : Colon(), 5)...]
        vw = @view w[ntuple(i -> i==4 ? wc : Colon(), 5)...]
        conv_im2col!(vy, vx, vw, new_diminfo)
    end
    
    return y
end