'torch'

    is_tensor
    is_storage
    is_complex
    is_conj
    is_floating_point
    is_nonzero
    set_default_dtype
    get_default_dtype
    set_default_device
    set_default_tensor_type
    numel
    set_printoptions
    set_flush_denormal

    tensor
    sparse_coo_tensor
    sparse_csr_tensor
    sparse_csc_tensor
    sparse_bsr_tensor
    sparse_bsc_tensor
    asarray
    as_tensor
    as_strided
    from_numpy
    from_dlpack
    frombuffer
    zeros
    zeros_like
    ones
    ones_like
    arange
    range
    linspace
    logspace
    eye
    empty
    empty_like
    empty_strided
    full
    full_like
    quantize_per_tensor
    quantize_per_channel
    dequantize
    complex
    polar
    heaviside

    adjoint     Returns a view of the tensor conjugated and with the last two dimensions transposed.
    argwhere    Returns a tensor containing the indices of all non-zero elements of input.
    cat         Concatenates the given sequence of seq tensors in the given dimension.
    concat      Alias of torch.cat().
    concatenate Alias of torch.cat().
    conj        Returns a view of input with a flipped conjugate bit.
    chunk       Attempts to split a tensor into the specified number of chunks.
    dsplit      Splits input, a tensor with three or more dimensions, into multiple tensors depthwise according to indices_or_sections.
    column_stack    Creates a new tensor by horizontally stacking the tensors in tensors.
    dstack      Stack tensors in sequence depthwise (along third axis).
    gather      Gathers values along an axis specified by dim.
    hsplit      Splits input, a tensor with one or more dimensions, into multiple tensors horizontally according to indices_or_sections.
    hstack      Stack tensors in sequence horizontally (column wise).
    index_add   See index_add_() for function description.
    index_copy  See index_add_() for function description.
    index_reduce    See index_reduce_() for function description.
    index_select    Returns a new tensor which indexes the input tensor along dimension dim using the entries in index which is a LongTensor.
    masked_select   Returns a new 1-D tensor which indexes the input tensor according to the boolean mask mask which is a BoolTensor.
    movedim     Moves the dimension(s) of input at the position(s) in source to the position(s) in destination.
    moveaxis    Alias for torch.movedim().
    narrow      Returns a new tensor that is a narrowed version of input tensor.
    narrow_copy Same as Tensor.narrow() except this returns a copy rather than shared storage.
    nonzero
    permute     Returns a view of the original tensor input with its dimensions permuted.
    reshape     Returns a tensor with the same data and number of elements as input, but with the specified shape.
    row_stack   Alias of torch.vstack().
    select      Slices the input tensor along the selected dimension at the given index.
    scatter     Out-of-place version of torch.Tensor.scatter_()
    diagonal_scatter    Embeds the values of the src tensor into input along the diagonal elements of input, with respect to dim1 and dim2.
    select_scatter      Embeds the values of the src tensor into input at the given index.
    slice_scatter       Embeds the values of the src tensor into input at the given dimension.
    scatter_add Out-of-place version of torch.Tensor.scatter_add_()
    scatter_reduce      Out-of-place version of torch.Tensor.scatter_reduce_()
    split       Splits the tensor into chunks.
    squeeze     Returns a tensor with all specified dimensions of input of size 1 removed.
    stack       Concatenates a sequence of tensors along a new dimension.
    swapaxes    Alias for torch.transpose().
    swapdims    Alias for torch.transpose().
    t           Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
    take        Returns a new tensor with the elements of input at the given indices.
    take_along_dim  Selects values from input at the 1-dimensional indices from indices along the given dim.
    tensor_split    Splits a tensor into multiple sub-tensors, all of which are views of input, along dimension dim according to the indices or number of sections specified by indices_or_sections.
    tile        Constructs a tensor by repeating the elements of input.
    transpose   Returns a tensor that is a transposed version of input.
    unbind      Removes a tensor dimension.
    unsqueeze   Returns a new tensor with a dimension of size one inserted at the specified position.
    vsplit      Splits input, a tensor with two or more dimensions, into multiple tensors vertically according to indices_or_sections.
    vstack      Stack tensors in sequence vertically (row wise).
    where       Return a tensor of elements selected from either input or other, depending on condition.


