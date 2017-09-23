proc hflip*(img: Tensor[uint8]): Tensor[uint8] {.inline.} =
  ## Horizontal flips an image
  result = img[_, _, ^1..0|-1]

proc vflip*(img: Tensor[uint8]): Tensor[uint8] {.inline.} =
  ## Vertical flips an image
  result = img[_, ^1..0|-1, _]

proc vhflip*(img: Tensor[uint8]): Tensor[uint8] {.inline.} =
  ## Flip vertically and horizontally an image
  result = img[_, ^1..0|-1, ^1..0|-1]

proc crop*(img: Tensor[uint8], x, y, width, height: int): Tensor[uint8] {.inline.} =
  ## Crop an image
  result = img[_, y..<(y+height), x..<(x+width)]

proc center_crop*[T](img: Tensor[T], width, height: int): Tensor[T] {.inline.} =
  ## Crop an image to center
  let
    x = (img.width - width) div 2
    y = (img.height - height) div 2
  result = img.crop(x, y, width, height)

proc random_crop*[T](img: Tensor[T], width, height: int): Tensor[T] {.inline.} =
  ## Random crop an image
  let
    x = random(img.width - width + 1)
    y = random(img.height - height + 1)
  result = img.crop(x, y, width, height)

proc rot90*[T](img: Tensor[T], k: int): Tensor[T] =
  ## Rotate an image 90 degrees clockwise `k` times
  case k mod 4:
    of 0:
      result = img
    of 1:
      result = img.permute(0,2,1).hflip
    of 2:
      result = img.vhflip
    else:
      result = img.permute(0,2,1).vflip

proc quantize_bytes*[T: SomeReal](img: Tensor[T], U: typedesc): Tensor[U] =
  ## Quantize image bytes, from type T to U, useful for converting
  ## images from floats to ints
  # TODO: nim bugs issue #6406 and #6407, ugly workaround solution:
  when U is uint8:
    result = img.map(proc(x: T): U = clamp(x + 0.5.T, low(U).T, high(U).T).uint8)
  else:
    static: assert false

proc quantize_bytes*[T: SomeInteger](img: Tensor[T], U: typedesc): Tensor[U] =
  ## Quantize image bytes, from type T to U, useful for converting
  ## images from floats to ints
  # TODO: nim bugs issue #6406 and #6407, ugly workaround solution:
  when U is uint8:
    img.map(proc(x: T): U = clamp(x, low(U).T, high(U).T).uint8)
  elif U is int:
    img.map(proc(x: T): U = clamp(x, low(U).T, high(U).T).int)
  else:
    static: assert false

type
  PadMode* = enum
    PadConstant = 0
    PadNearest = 1

proc im2col*[T](input: Tensor[T], ksize: int, pad: int = 0, mode: PadMode, pad_constant: int): Tensor[T] =
  ## Convert blocks of an image into columns, useful for preprocessing
  ## an image before convolutions, pad mode.
  let
    channels = input.channels
    height = input.height
    width = input.width
    channels_col = channels * ksize * ksize
    height_col = height + (2 * pad) - ksize + 1
    width_col = width + (2 * pad) - ksize + 1
  result = zeros([channels_col, height_col * width_col], T)
  for c in 0..<channels_col:
    let
      w_offset = (c mod ksize) - pad
      h_offset = ((c div ksize) mod ksize) - pad
      c_offset = (c div ksize) div ksize
    for h in 0..<height_col:
      let
        row = h_offset + h
        offset_col = h * width_col
      for w in 0..<width_col:
        let col = w_offset + w
        if row < 0 or col < 0 or row >= height or col >= width:
          case mode:
            of PadConstant:
              if pad_constant != 0:
                result[c, offset_col + w] = pad_constant
            of PadNearest:
              result[c, offset_col + w] = input[c_offset, clamp(row, 0, height-1), clamp(col, 0, width-1)]
        else:
          result[c, offset_col + w] = input[c_offset, row, col]

proc correlate2d*[T,U](input: Tensor[T], weights: Tensor[U], pad: int = 0, mode: PadMode = PadConstant, cval: U = 0): Tensor[int] =
  ## Correlate an image with the given kernel weights, this is a convolution
  ## without flipping the kernel
  let ksize = weights.width

  assert input.rank == 3
  assert weights.rank == 3
  assert weights.width == weights.height
  assert ksize > 0 and ksize mod 2 == 1

  let
    channels = input.channels
    height = input.height + (2 * pad) - ksize + 1
    width = input.width + (2 * pad) - ksize + 1
    channel_ksize = ksize*ksize

  var w = weights.reshape([channels, 1, ksize*ksize])
  var x = im2col(input.astype(U), ksize, pad, mode, cval).unsafeReshape([channels, channel_ksize, height*width])
  var res_channels = newSeq[Tensor[U]](channels)

  for c in 0..<channels:
    res_channels[c] = (w.unsafeAt(c) * x.unsafeAt(c)).unsafeReshape([1, height, width])

  result = concat(res_channels, 0)

proc convolve2d*(input: Tensor[uint8], weights: Tensor[int], pad: int, mode: PadMode = PadConstant, cval: int = 0): Tensor[int] =
  ## Convolve an image with the given kernel weights, like correlate but
  ## it flips the kernel before.
  let flipped_weights = weights.unsafeView(_, ^1..0|-1, ^1..0|-1)
  result = correlate2d(input, flipped_weights, pad, mode, cval)

proc tile_collection*(imgs: Tensor[uint8], max_width: int = 0): Tensor[uint8] =
  assert imgs.rank == 4
  let
    count = imgs.shape[0]
    cell_width = imgs.width
    cell_height = imgs.height
  var cols : int
  if max_width == 0:
    cols = ceil(sqrt(count.float)).int
  else:
    cols = max_width div cell_width
  var rows = ceil(count / cols).int

  result = zeros([imgs.channels, rows*cell_height, cols*cell_width], uint8)
  for i in 0..<count:
    let
      y = (i div cols) * cell_height
      x = (i mod cols) * cell_width
    result[_, y..<(y+cell_height), x..<(x+cell_width)] = imgs[i, _, _, _].unsafeSqueeze(0)

