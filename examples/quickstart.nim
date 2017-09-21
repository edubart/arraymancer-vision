import arraymancer_vision

# Load image from file into a CxHxW Tensor[uint8]
var image = load("examples/lena.png")

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
