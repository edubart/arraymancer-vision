type
  ScaleMode* = enum
    ScaleNearest = 0
    ScaleBilinear = 1

proc round_pixel(a: float32, U: typedesc): U {.inline.} =
  when U is uint8:
    clamp(a + 0.5.float32, low(U).float32, high(U).float32).uint8
  elif U is float32:
    a.float32

proc scale_nearest[T](src: Tensor[T], width, height: int): Tensor[T] {.inline.} =
  result = newTensor[T]([src.channels, height, width])
  let
    step_x = src.height.float32 / height.float32
    step_y = src.width.float32 / width.float32
  for c in 0..<src.channels:
    for y in 0..<height:
      let sy = (y.float32 * step_y).int
      for x in 0..<width:
        let sx = (x.float32 * step_x).int
        result[c, y, x] = src[c, sy, sx]

proc scale_linear_vertical[T](src: Tensor[T], width, height: int): Tensor[T] {.inline.} =
  result = newTensor[T]([src.channels, height, width])
  let
    step_y = src.height.float32 / height.float32
    max_sy = src.height - 1
  for c in 0..<src.channels:
    for y in 0..<height:
      let
        sy = y.float32 * step_y
        say = sy.int
        sby = min(say+1, max_sy)
        sa_factor = sby.float32 - sy
        sb_factor = 1.0f - sa_factor
      for x in 0..<width:
        let
          sx = x
          sa = src[c, say, sx].float32 * sa_factor
          sb = src[c, sby, sx].float32 * sb_factor
        result[c, y, x] = round_pixel(sa + sb, T)

proc scale_linear_horizontal[T](src: Tensor[T], width, height: int): Tensor[T] {.inline.} =
  result = newTensor[T]([src.channels, height, width])
  let
    step_x = src.width.float32 / width.float32
    max_sx = src.width - 1
  for c in 0..<src.channels:
    for y in 0..<height:
      let sy = y
      for x in 0..<width:
        let
          sx = x.float32 * step_x
          sax = sx.int
          sbx = min(sax+1, max_sx)
          sa_factor = sbx.float32 - sx
          sb_factor = 1.0f - sa_factor
          sa = src[c, sy, sax].float32 * sa_factor
          sb = src[c, sy, sbx].float32 * sb_factor
        result[c, y, x] = round_pixel(sa + sb, T)

proc scale_bilinear[T](src: Tensor[T], width, height: int): Tensor[T] {.inline.} =
  var tmp : Tensor[T]
  if height != src.height:
    tmp = scale_linear_vertical(src, src.width, height)
  else:
    shallowCopy(tmp, src)
  if width != src.width:
    result = scale_linear_horizontal(tmp, width, height)
  else:
    result = tmp

proc scale*[T](src: Tensor[T], width, height: int, mode: ScaleMode = ScaleNearest): Tensor[T] =
  ## Scale an image to a new size, suppored modes are nearest, and bilinear,
  ## defaults to nearest.
  case mode:
    of ScaleNearest:
      return scale_nearest(src, width, height)
    of ScaleBilinear:
      return scale_bilinear(src, width, height)
