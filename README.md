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
var image = load("assets/lena.png")

# Do some preprocessing
image = image.center_crop(128, 128)
image = image.hflip()
image = image.rot90(1)
image = image.filter_sharpen()
image = image.scale(256, 256, ScaleBilinear)

# Visualize it using visdom
let vis = newVisdomClient()
vis.image(image)

# Save it to a file
image.save("preprocessed_lena.png")
```

## API

Documentation of the completely available API you read [here](https://rawgit.com/edubart/arraymancer-vision/master/doc/documentation.html)

### Details

The library operates all images as Tensor[uint8] with dimensions CxHxW, where C is in RGBA colorspace, note that other image libraries usually operates with images in HxWxC format, so remember this when using. This design choice is to optimize and facilitate operation on images in deep learning tasks.

## TODO

* Loading multiple images
* Simple drawing routines
* Colorspace conversions
