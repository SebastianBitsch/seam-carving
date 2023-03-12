import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from gradientmagniture import SobelGradientMagnitude, ScharrGradientMagnitude, ExactGradientMagnitude
from seamcarve import SeamCarver


def plot(im, size = (10,10), cbar=False, cmap="viridis", title="Image"):
    height, width = im.shape[:2]
    height *= 2
    width *= 2
    margin=200 # pixels
    dpi=100. # dots per inch

    figsize=((width+4*margin)/dpi, (height+2*margin)/dpi) # inches
    left = margin/dpi/figsize[0] #axes ratio
    bottom = margin/dpi/figsize[1]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.title(title, loc="left")

    fig.subplots_adjust(left=left, bottom=bottom, right=1.-3*left, top=1.-bottom)
        
    plt.imshow(im, cmap=cmap)
    if cbar:
        cbar_ax = fig.add_axes([1 - 3*left, bottom, 0.01, 1.-2*bottom])
        plt.colorbar(cax=cbar_ax)
    plt.show()


def plot_demo(im):

    grad = ExactGradientMagnitude()
    carver = SeamCarver(im, grad)

    num_seams = 100
    carver.carve_seams(vertically = True, num_seams=num_seams)

    # Plot original image
    plot(carver.im, title = "1: Original image")
    plot(carver.im_gray, title = "2: Convert to gray scale", cmap="gray")
    plot(carver.im_grad, title = "3: Compute the gradient magnitude per pixel", cbar=True, cmap="gray")
    plot(carver.cumulative, title = "4: Sum minimum gradients from the bottom up", cbar=True, cmap="gray")
    # Draw seams
    plot(carver.draw_seams(carver.cumulative, 1), title=f"5: Find cheapest path from bottom to top using dynamic programming")
    for i in range(5, len(carver.seams), 5):
        plot(carver.draw_seams(carver.cumulative, i), title=f"5: Find ({i}) cheapest paths from bottom to top using dynamic programming")
    plot(carver.draw_seams(carver.cumulative, len(carver.seams)), title=f"5: Find ({len(carver.seams)}) cheapest paths from bottom to top using dynamic programming")
    seams_im = carver.draw_seams(carver.im, i)
    plot(seams_im, title=f"6: All seams carved")

    # Plot Removed slices ontop cummulative image
    for i in [0.25, 0.2, 0.15, 0.1]:
        res1 = carver.remove_seams(carver.im, carve_threshold=i)
        res2 = carver.remove_seams(seams_im, carve_threshold=i)
        im_width = carver.im.shape[1]
        im_width_new = res1.shape[1]
        plot(res2, title = f"7: Shink image by removing columns with {int(i*100)}% seams ({im_width - im_width_new}px removed)")
        plot(res1, title = f"7: Shink image by removing columns with {int(i*100)}% seams ({im_width - im_width_new}px removed)")
    

if __name__ == "__main__":
    im = Image.open("images/tower.png")

    # 1. Plot for demo
    plot_demo(im)


    # 2. Use as library
    grad = ExactGradientMagnitude()
    carver = SeamCarver(im, grad)

    res = carver.compress()
    plot(res)

    