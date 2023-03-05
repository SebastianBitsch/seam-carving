import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from collections import defaultdict


from gradientmagniture import GradientMagnitude

class SeamCarver:

    def __init__(self, image: Image, grad_func: GradientMagnitude) -> None:
        self.im = np.asarray(image)
        self.im_gray = np.asarray(image.convert("L"))
        self.grad_func = grad_func
        
    
    def _calc_cumulative_min_energy(self, im_grad:np.ndarray, vertically = True) -> np.ndarray:
        """ Calculates the cumulative energy in an image either horizontally or vertically """
        if not vertically:
            im_grad = im_grad.T
            
        cumulative = np.copy(im_grad)

        for i in range(1, im_grad.shape[0]):
            for j in range(0, im_grad.shape[1]):
                offset = 0 if j == 0 else 1 # TODO: not sure if this is needed
                cumulative[i,j] = np.min(cumulative[i-1,j-offset:j+2]) + im_grad[i,j]
        
        if not vertically:
            cumulative = cumulative.T
        return cumulative

    
    def carve(self, vertically = True, num_seams:int = 200, carve_threshold: float = 0.3):
        self.im_grad = self.grad_func.grad(self.im_gray)

        self.cumulative = self._calc_cumulative_min_energy(self.im_grad, vertically)

        self.seams = self.create_seams(self.cumulative, num_seams, vertically)

        seams = np.array(self.seams)

        num = self.im.shape[1] if vertically else self.im.shape[0]
        mask = [carve_threshold < ((seams == i).sum() / num) for i in range(num)]
        
        return np.delete(self.im, mask, axis=vertically)

    
    def create_seams(self, cumulative:np.ndarray, num_seams:int, vertically = True) -> list[list]:
        if not vertically:
            cumulative = cumulative.T
        
        h, w = cumulative.shape

        seams = []

        positions_filled = defaultdict(set)

        for k in range(num_seams):

            start_index = np.argsort(cumulative[-1,:])[k]
            
            seam = [start_index]
            cumulative[h-1, start_index] = 10000

            positions_filled[h-1].add(start_index)

            for i in range(h-1, 0, -1):
                
                current = seam[-1]
                
                best_option = np.Inf
                best_option_index = None

                for opt in range(current-1, current+2):

                    if (opt < 0) or (w - 1 < opt) or (opt in positions_filled[i-1]):
                        continue
                
                    if cumulative[i-1, opt] < best_option:
                        best_option = cumulative[i-1, opt]
                        best_option_index = opt
                        
                if best_option_index == None:
                    best_option_index = current

                positions_filled[i-1].add(best_option_index)
                seam.append(best_option_index)            
            seams.append(seam)
        
        return seams
