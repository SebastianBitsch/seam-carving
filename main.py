import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from collections import defaultdict

from gradientmagniture import SobelGradientMagnitude, ScharrGradientMagnitude, ExactGradientMagnitude
from seamcarve import SeamCarver


def plot(im, size = (10,10), cbar=False, cmap="viridis", title="Image"):
    height, width = im.shape[:2]
    margin=200 # pixels
    dpi=100. # dots per inch

    # fig, ax = plt.subplots(dpi=dpi)#figsize=size)

    figsize=((width+4*margin)/dpi, (height+2*margin)/dpi) # inches
    left = margin/dpi/figsize[0] #axes ratio
    bottom = margin/dpi/figsize[1]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.title(title, loc="left")

    fig.subplots_adjust(left=left, bottom=bottom, right=1.-3*left, top=1.-bottom)
    
    
    # ax.set_title(title)
    # im = ax.imshow(im, interpolation='none', cmap=cmap)
    # fig.figimage(im)
    # plt.tight_layout()
    
    plt.imshow(im, cmap=cmap)
    if cbar:
        cbar_ax = fig.add_axes([1 - 3*left, bottom, 0.01, 1.-2*bottom])
        # new ax with dimensions of the colorbar
        # cbar = fig.colorbar(im, cax=cbar_ax)
        plt.colorbar(cax=cbar_ax)
    plt.show()

if __name__ == "__main__":

    im = Image.open("images/tower.png")

    grad = ExactGradientMagnitude()
    carver = SeamCarver(im, grad)


    num_seams = 100
    seams = carver.carve_seams(vertically = True, num_seams=num_seams)

    # Plot original image
    plot(carver.im, title = "1: Original image")
    plot(carver.im_gray, title = "2: Convert to gray scale", cmap="gray")
    plot(carver.im_grad, title = "3: Compute the gradient per pixel", cbar=True, cmap="gray")
    plot(carver.cumulative, title = "4: Sum gradients from the bottom up", cbar=True, cmap="gray")
    # Draw seams
    for i in range(1, len(carver.seams)):
        plot(carver.draw_seams(carver.cumulative, i), title=f"5: Find ({i}) cheapest paths using DP")
    seams_im = carver.draw_seams(carver.im, i)
    plot(seams_im, title=f"6: All seams carved")

    # Plot Removed slices ontop cummulative image
    for i in [0.3, 0.2, 0.15, 0.1]:
        res1 = carver.remove_seams(carver.im, carve_threshold=i)
        res2 = carver.remove_seams(seams_im, carve_threshold=i)
        im_width = carver.im.shape[1]
        im_width_new = res1.shape[1]
        plot(res2, title = f"Shink image by removing seams ({im_width - im_width_new}px removed)")
        plot(res1, title = f"Shink image by removing seams ({im_width - im_width_new}px removed)")
        
        
    

    # plot(carver.im_grad)
    # plot(carver.cumulative)
    # plot(carver.draw_seams(len(carver.seams)))
