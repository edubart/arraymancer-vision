proc kernel*[T,U](img: Tensor[T], kernel: Tensor[U], scale: U = 1, offset: U = 0): Tensor[T] =
  ## Applies a kernel matrix to an image and divides the outputs by scale factor
  ## and them sun offset.
  ## For more information see https://en.wikipedia.org/wiki/Kernel_(image_processing)
  ##
  ## Implementation details:
  ## This functions does not flip the kernel, so it does image correlation
  ## instead of convolution.
  ## The padding borders of the image is replaced with the nearest neighbourhood
  ## border.

  assert kernel.width == kernel.height
  let kernel = kernel.bc([img.channels, kernel.height, kernel.width])
  let pad = (kernel.width - 1) div 2
  var correlated_img = img.correlate2d(kernel, pad, PadNearest)
  if scale != 1.U:
    correlated_img /= scale
  if offset != 0.U:
    correlated_img += offset.bc(correlated_img.shape)
  result = quantize_bytes(correlated_img, T)

proc filter_blur*[T](img: Tensor[T]): Tensor[T] =
  ## Blur an image using a predefied kernel
  img.kernel([[
    [1,  1,  1,  1,  1],
    [1,  0,  0,  0,  1],
    [1,  0,  0,  0,  1],
    [1,  0,  0,  0,  1],
    [1,  1,  1,  1,  1]
  ]].toTensor(), 16)

proc filter_contour*[T](img: Tensor[T]): Tensor[T] =
  ## Contour an image using a predefied kernel
  img.kernel([[
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
  ]].toTensor(), 1, 255)

proc filter_detail*[T](img: Tensor[T]): Tensor[T] =
  ## Detail an image using a predefied kernel
  img.kernel([[
    [ 0,  -1,  0],
    [-1,  10, -1],
    [ 0,  -1,  0],
  ]].toTensor(), 6)

proc filter_edge_enhance*[T](img: Tensor[T]): Tensor[T] =
  ## Enhance edges of an image using a predefied kernel
  img.kernel([[
    [-1,  -1, -1],
    [-1,  10, -1],
    [-1,  -1, -1],
  ]].toTensor(), 2)

proc filter_edge_enhance_more*[T](img: Tensor[T]): Tensor[T] =
  ## Enhance edges of an image using a predefied kernel
  img.kernel([[
    [-1,  -1, -1],
    [-1,   9, -1],
    [-1,  -1, -1],
  ]].toTensor(), 1)

proc filter_emboss*[T](img: Tensor[T]): Tensor[T] =
  ## Enhance an image using a predefied kernel
  img.kernel([[
    [-1,  0, 0],
    [0,  1, 0],
    [0,  0, 0],
  ]].toTensor(), 1, 128)

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
