# Arraymancer Vision (WIP)

Simple library for image loading, preprocessing and visualization for working with arraymancer.

## Features

* Loading image into tensors
* Simple image transformations like flipping, rotation, scaling
* Saving images
* Image convolution  filters like sharpen, edges
* Visualization of images using [visdom](https://github.com/facebookresearch/visdom)

# Quick Start

## Installation

Install using nimble package manager:

```Bash
nimble install arraymancer-vision
```

For visualizing you have to install visdom and run it:

```Bash
pip install visdom
python -m visdom.server
```

Then go to http://localhost:8097

## Usage example

```Nim
import arraymancer_vision

# Load image from file into a CxHxW Tensor[uint8]
var origimage = load("examples/lena.png")

# Do some preprocessing
var image = origimage.center_crop(128, 128)
image = image.hflip()
image = image.rot90(1)
image = image.filter_sharpen()
image = image.scale(512, 512, ScaleBilinear)

# Visualize it using visdom
let vis = newVisdomClient()
vis.image(origimage)
vis.image(image)

# Save it to a file
image.save("examples/preprocessed_lena.png")
```

This quickstart example is inside examples directory, you can run it by
cloning the repo and running with `nim c -r examples/quickstart.nim`

You can visualize all predefined filters having visdom running and then
running the filters example with `nim c -r examples/visualize_filters.nim`

## API

Documentation of the completely available API is [here](https://rawgit.com/edubart/arraymancer-vision/master/doc/documentation.html)

## Details

The library operates all images as `Tensor[uint8]` with dimensions CxHxW, where C is in RGBA colorspace, note that other image libraries usually operates with images in HxWxC format, so remember this when using. This design choice is to optimize and facilitate operation on images in deep learning tasks.

## TODO

* Loading multiple images
* Simple drawing routines
* Colorspace conversions
