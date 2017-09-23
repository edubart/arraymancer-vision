import arraymancer_vision

# Load image from file into a CxHxW Tensor[uint8]
var image = load("examples/lena.png")

# Visualize filters on it using visdom
let vis = newVisdomClient()
vis.image(image, "Normal")
vis.image(image.filter_blur(), "Blur")
vis.image(image.filter_contour(), "Contour")
vis.image(image.filter_detail(), "Detail")
vis.image(image.filter_edge_enhance(), "Edge Enhance")
vis.image(image.filter_edge_enhance_more(), "Edge Enhance More")
vis.image(image.filter_emboss(), "Emboss")
vis.image(image.filter_smooth(), "Smooth")
vis.image(image.filter_smooth_more(), "Smooth More")
vis.image(image.filter_sharpen(), "Sharpen")
vis.image(image.filter_find_edges(), "Find Edges")
