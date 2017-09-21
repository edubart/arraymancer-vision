import arraymancer
import ../src/arraymancer_vision
import unittest, sequtils

suite "Image IO":
  let imga = [[
    [0'u8, 1'u8],
    [2'u8, 3'u8],
    [4'u8, 5'u8]
  ]].toTensor()

  test "Image flipping":
    check imga.vflip == [[
      [4'u8, 5'u8],
      [2'u8, 3'u8],
      [0'u8, 1'u8]
    ]].toTensor()

    check imga.hflip == [[
      [1'u8, 0'u8],
      [3'u8, 2'u8],
      [5'u8, 4'u8]
    ]].toTensor()

    check imga.vhflip == [[
      [5'u8, 4'u8],
      [3'u8, 2'u8],
      [1'u8, 0'u8]
    ]].toTensor()

  test "Image rotation":
    check imga.rot90(0) == [[
      [0'u8, 1'u8],
      [2'u8, 3'u8],
      [4'u8, 5'u8]
    ]].toTensor()

    check imga.rot90(1) == [[
      [4'u8, 2'u8, 0'u8],
      [5'u8, 3'u8, 1'u8]
    ]].toTensor()

    check imga.rot90(2) == [[
      [5'u8, 4'u8],
      [3'u8, 2'u8],
      [1'u8, 0'u8]
    ]].toTensor()

    check imga.rot90(3) == [[
      [1'u8, 3'u8, 5'u8],
      [0'u8, 2'u8, 4'u8]
    ]].toTensor()

  test "Image cropping":
    check imga.crop(0, 0, 2, 2) == [[
      [0'u8, 1'u8],
      [2'u8, 3'u8]
    ]].toTensor()
    check imga.crop(x=0, y=0, width=imga.width, height=imga.height) == imga

    check imga.center_crop(width=2, height=1) == [[
      [2'u8, 3'u8]
    ]].toTensor()
    check imga.center_crop(width=1, height=2) == [[
      [0'u8],
      [2'u8]
    ]].toTensor()

  test "Quantize bytes":
    block:
      let a = [[
        [1.0, 255.0, 1.5],
        [3.1, 5.9, 258],
        [-2.0, -0.1, 9.6],
      ]].toTensor()

      check a.quantize_bytes(uint8) == [[
        [1'u8, 255, 2],
        [3'u8, 6, 255],
        [0'u8, 0, 10],
      ]].toTensor()

    block:
      let a = [[
        [1, 255, 1],
        [3, 0, 258],
        [-2, -1, 9],
      ]].toTensor()

      check a.quantize_bytes(uint8) == [[
        [1'u8, 255, 1],
        [3'u8, 0, 255],
        [0'u8, 0, 9],
      ]].toTensor()

  test "Image convolve2d":
    let a =
     [[[1'u8, 2, 0, 0],
       [5'u8, 3, 0, 4],
       [0'u8, 0, 0, 7],
       [9'u8, 3, 0, 0]]].toTensor()

    let k =
     [[[1, 1, 1],
       [1, 1, 0],
       [1, 0, 0]]].toTensor()

    check: a.convolve2d(k, 1) ==
     [[[11, 10,  7,  4],
       [10,  3, 11, 11],
       [15, 12, 14,  7],
       [12,  3,  7,  0]]].toTensor()

    check: a.convolve2d(k, 1, PadConstant, 1) ==
     [[[13, 11,  8,  7],
       [11,  3, 11, 14],
       [16, 12, 14, 10],
       [15,  6, 10,  5]]].toTensor()

  test "Image filters":
    let a =
     [[[1'u8, 2, 0, 0],
       [5'u8, 3, 0, 4],
       [0'u8, 0, 0, 7],
       [9'u8, 3, 0, 0]]].toTensor()

    let b = a.filter_sharpen()
    check: b ==
     [[[ 0'u8,  2,  0,  0],
       [ 8'u8,  5,  0,  5],
       [ 0'u8,  0,  0,  12],
       [13'u8,  3,  0,  0]]].toTensor()

  test "Scale nearest":
    let a = [[
      [ 0'u8,  32],
      [64'u8, 128],
    ]].toTensor()

    check a.scale(4, 4, ScaleNearest) == [[
      [ 0'u8,  0,  32,  32],
      [ 0'u8,  0,  32,  32],
      [64'u8, 64, 128, 128],
      [64'u8, 64, 128, 128],
    ]].toTensor()

  test "Image load and save":
    let lena = vision.load("examples/lena.png")
    let lena2 = vision.loadFromMemory(lena.toBMP())
    check: lena2 == lena

    expect(IOError):
      discard vision.load("inexhistent_image_file.png")
