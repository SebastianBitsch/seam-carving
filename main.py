import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from collections import defaultdict

from gradientmagniture import SobelGradientMagnitude, ScharrGradientMagnitude, ExactGradientMagnitude
from seamcarve import SeamCarver


def plot(im, size = (12,12), cbar=False, cmap="viridis"):
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(im, interpolation='nearest', cmap=cmap)
    plt.tight_layout()
    if cbar:
        fig.colorbar(im, ax = ax)
    plt.show()

if __name__ == "__main__":

    im = Image.open("images/tower.png")

    grad = ScharrGradientMagnitude()
    carver = SeamCarver(im, grad)

    plot(carver.carve(vertically = True, num_seams=100, carve_threshold=0.5))