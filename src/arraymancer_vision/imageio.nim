proc channels*[T](img: Tensor[T]): int {.inline.}  =
  ## Return number of channels of the image
  img.shape[^3]

proc height*[T](img: Tensor[T]): int {.inline.} =
  ## Return height of the image
  img.shape[^2]

proc width*[T](img: Tensor[T]): int {.inline.}  =
  ## Return width of the image
  img.shape[^1]

proc hwc2chw*[T](img: Tensor[T]): Tensor[T] =
  ## Convert image from HxWxC convetion to the CxHxW convention,
  ## where C,W,H stands for channels, width, height, note that this library
  ## only works with CxHxW images for optimization and internal usage reasons
  ## using CxHxW for images is also a common approach in deep learning
  img.permute(2, 0, 1)

proc chw2hwc*[T](img: Tensor[T]): Tensor[T] =
  ## Convert image from CxHxW convetion to the HxWxC convention,
  ## where C,W,H stands for channels, width, height, note that this library
  ## only works with CxHxW images for optimization and internal usage reasons
  ## using CxHxW for images is also a common approach in deep learning
  img.permute(1, 2, 0)

proc pixels*(img: Tensor[uint8]): seq[uint8] =
  # Return contiguous pixel data in the HxWxC convetion, method intended
  # to use for interfacing with other libraries
  img.chw2hwc().asContiguous().data

proc load*(filename: string, desired_channels: int = 0): Tensor[uint8] =
  ## Load image from file, with the desired number of channels,
  ## into a contiguous CxHxW Tensor[uint8]. Desired channels defaults to 0 meaning
  ## that it will auto detect the number of channels, the returned image tensor
  ## will be in the CxHxW format even for images with a single channel.
  ##
  ## Supports PNG, JPG, BMP, TGA and HDR formats
  ##
  ## On error an IOError exception will be thrown
  var width, height, channels: int
  try:
    let pixels = stbi.load(filename, width, height, channels, desired_channels)
    result = pixels.unsafeToTensorReshape([height, width, channels]).hwc2chw().asContiguous()
    assert(desired_channels == 0 or channels == desired_channels)
  except STBIException:
    raise newException(IOError, getCurrentExceptionMsg())

proc loadFromMemory*(contents: string, desired_channels: int = 0): Tensor[uint8] =
  ## Like load but loads from memory, the contents must be a buffer
  ## for a supported image format
  var width, height, channels: int
  let pixels = stbi.loadFromMemory(cast[seq[uint8]](toSeq(contents.items)), width, height, channels, desired_channels)
  result = pixels.unsafeToTensorReshape([height, width, channels]).hwc2chw().asContiguous()
  assert(desired_channels == 0 or channels == desired_channels)

proc save*(img: Tensor[uint8], filename: string, jpeg_quality: int = 100) =
  ## Save an image to a file, supports PNG, BMP, TGA and JPG.
  ## Argument `jpeg_quality` can be passed to inform the saving
  ## quality from a range 0 to 100, defaults to 100
  var ok = false
  if filename.endsWith(".png"):
    ok = stbiw.writePNG(filename, img.width, img.height, img.channels, img.pixels)
  elif filename.endsWith(".bmp"):
    ok = stbiw.writeBMP(filename, img.width, img.height, img.channels, img.pixels)
  elif filename.endsWith(".tga"):
    ok = stbiw.writeTGA(filename, img.width, img.height, img.channels, img.pixels)
  elif filename.endsWith(".jpg"):
    ok = stbiw.writeJPG(filename, img.width, img.height, img.channels, img.pixels, jpeg_quality)

  if not ok:
    raise newException(IOError, "Failed to save image to a file: " & filename)

proc toPNG*(img: Tensor[uint8]): string =
  ## Convert an image to PNG into a string of bytes
  return stbiw.writePNG(img.width, img.height, img.channels, img.pixels)

proc toBMP*(img: Tensor[uint8]): string =
  ## Convert an image to BMP into a string of bytes
  return stbiw.writeBMP(img.width, img.height, img.channels, img.pixels)

proc toTGA*(img: Tensor[uint8]): string =
  ## Convert an image to TGA into a string of bytes
  return stbiw.writeTGA(img.width, img.height, img.channels, img.pixels)

proc toJPG*(img: Tensor[uint8], quality: int = 100): string =
  ## Convert an image to JPG into a string of bytes.
  ## Argument `jpeg_quality` can be passed to inform the saving
  ## quality from a range 0 to 100, defaults to 100
  return stbiw.writeJPG(img.width, img.height, img.channels, img.pixels, quality)
