from abc import ABC, abstractmethod

import numpy as np

class GradientMagnitude(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def grad(self, image: np.ndarray) -> np.ndarray:
        pass
    
    def conv2d(self, img: np.ndarray, krn:np.ndarray) -> np.ndarray:
        """ Compute 2d convolution, stolen from stackoverflow """
        is0, is1, ks0, ks1 = *img.shape, *krn.shape
        rs0, rs1 = is0 - ks0 + 1, is1 - ks1 + 1
        
        ix0 = np.arange(ks0)[:, None] + np.arange(rs0)[None, :]
        ix1 = np.arange(ks1)[:, None] + np.arange(rs1)[None, :]
        
        res = krn[:, None, :, None] * img[(ix0.ravel()[:, None], ix1.ravel()[None, :])].reshape(ks0, rs0, ks1, rs1)
        res = res.transpose(1, 3, 0, 2).reshape(rs0, rs1, -1).sum(axis = -1)
        
        return res
    
    def pad(self, im:np.ndarray) -> np.ndarray:
        return np.pad(im, 1, 'edge')


class ExactGradientMagnitude(GradientMagnitude):

    def __init__(self) -> None:
        super().__init__()

    def grad(self, image: np.ndarray) -> np.ndarray:
        """
        Computes the gradient magnitude. Not very fast but ok
        See: https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/
        """
        # The output image will be one pixel smaller on every edge
        h, w = image.shape
        out = np.zeros((h - 2, w - 2))

        for y in range(1, w - 1):
            for x in range(1, h - 1):
                gx = int(image[x - 1, y]) - int(image[x + 1, y])
                gy = int(image[x, y - 1]) - int(image[x, y + 1])
                g = np.sqrt(gx ** 2 + gy ** 2)
                out[x - 1, y - 1] = g

        return self.pad(out)


class SobelGradientMagnitude(GradientMagnitude):

    def __init__(self) -> None:
        super().__init__()
        self.gx_sobel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ])
        self.gy_sobel = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ])
    
    def grad(self, image: np.ndarray) -> np.ndarray:
        """ https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/ """
        
        gx = self.conv2d(image, self.gx_sobel)
        gy = self.conv2d(image, self.gy_sobel)
        out = np.sqrt(np.power(gx,2) + np.power(gy,2))
        return self.pad(out)


class ScharrGradientMagnitude(GradientMagnitude):

    def __init__(self) -> None:
        super().__init__()
        self.gx_scharr = np.array([
            [3, 0, -3],
            [10, 0, -10],
            [3, 0, -3],
        ])
        self.gy_scharr = np.array([
            [3, 10, 3],
            [0, 0, 0],
            [-3, -10, -3],
        ])
    
    def grad(self, image: np.ndarray) -> np.ndarray:
        """ https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/ """
        
        gx = self.conv2d(image, self.gx_scharr)
        gy = self.conv2d(image, self.gy_scharr)
        out = np.sqrt(np.power(gx,2) + np.power(gy,2))
        return self.pad(out)
