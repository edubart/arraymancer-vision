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
