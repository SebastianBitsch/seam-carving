from PIL import Image
import matplotlib.pyplot as plt

from gradientmagniture import SobelGradientMagnitude, ScharrGradientMagnitude, ExactGradientMagnitude
from seamcarve import SeamCarver


if __name__ == "__main__":

    im = Image.open("images/waterfall.jpg")
    grad = ExactGradientMagnitude()

    carver = SeamCarver(im, grad)

    small_im = carver.carve(n_seams = 50)

    plt.imshow(small_im, interpolation="none")
    plt.show()


    