proc kernel*[T,U](img: Tensor[T], kernel: Tensor[U], scale: U = 1): Tensor[T] =
  ## Applies a kernel matrix to an image and divides the outputs by scale factor,
  ## which defaults to 1.
  ## For more information see https://en.wikipedia.org/wiki/Kernel_(image_processing)
  ##
  ## Implementation defailts:
  ## This functions does not flip the kernel, so it does image correlation
  ## instead of convolution.
  ## The padding borders of the image is replaced with the nearest neighbourhood
  ## border.

  assert kernel.width == kernel.height
  let kernel = kernel.bc([img.channels, kernel.height, kernel.width])
  let pad = (kernel.width - 1) div 2
  var correlated_img = img.correlate2d(kernel, pad, PadNearest)
  if scale != 1:
    correlated_img /= scale
  result = quantize_bytes(correlated_img, T)

proc filter_sharpen*[T](img: Tensor[T]): Tensor[T] =
  ## Sharpen an image using a predefied kernel
  img.kernel([[
    [-2,  -2, -2],
    [-2,  32, -2],
    [-2,  -2, -2],
  ]].toTensor(), 16)

proc filter_smooth*[T](img: Tensor[T]): Tensor[T] =
  ## Smooth an image using a predefied kernel
  img.kernel([[
    [1,  1,  1],
    [1,  5,  1],
    [1,  1,  1],
  ]].toTensor(), 13)

proc filter_smooth_more*[T](img: Tensor[T]): Tensor[T] =
  ## Smooth more an image using a predefied kernel
  img.kernel([[
    [1,  1,  1,  1,  1],
    [1,  5,  5,  5,  1],
    [1,  5, 44,  5,  1],
    [1,  5,  5,  5,  1],
    [1,  1,  1,  1,  1]
  ]].toTensor(), 100)

proc filter_find_edges*[T](img: Tensor[T]): Tensor[T] =
  ## Find edges of image using a predefied kernel
  img.kernel([[
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
  ]].toTensor(), 1)
