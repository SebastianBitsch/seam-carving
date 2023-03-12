import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from collections import defaultdict

from gradientmagniture import GradientMagnitude

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


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


    def draw_seams(self, ontop, n=1):
        im = ontop.copy()
        if len(im.shape) == 2:
            im = (255*normalize(np.stack((im,)*3, axis=-1))).astype(int)

        red = np.array([255,0,0])
        for i in range(n):
            im[np.arange(0, im.shape[0]), self.seams[i],:] = red
        return im

    def carve_seams(self, vertically:bool = True, num_seams:int = np.inf):
        self.im_grad = self.grad_func.grad(self.im_gray)

        self.cumulative = self._calc_cumulative_min_energy(self.im_grad, vertically)

        self.seams = self.create_seams(self.cumulative, num_seams, vertically)
        return self.seams

    def compress(self, vertically:bool=True, threshold_pct:float = 0.25):
        self.carve_seams(vertically)
        return self.remove_seams(self.im, threshold_pct, vertically)

    def remove_seams(self, fromim, carve_threshold:float = 0.1, vertically:bool = True):
        seams = np.array(self.seams)

        num = fromim.shape[1] if vertically else fromim.shape[0]
        mask = [carve_threshold < ((seams == i).sum() / num) for i in range(num)]
        
        return np.delete(fromim, mask, axis=vertically)

    def create_seams(self, cumulative:np.ndarray, num_seams:int, vertically = True) -> list[list]:
        seams = []
        
        positions_filled = defaultdict(set)
        h, w = cumulative.shape
        k = 0

        while len(seams) < num_seams and k < w:
            
            # Start the seam at the k largest value in the bottom row
            start_index = np.argsort(cumulative[-1,:])[k] 

            # Will be a h length list of width-indices from the bottom up
            seam = [start_index]
            seam_cost = 0

            # Pick the next index
            for i in range(h-2, -1, -1):
                last = seam[-1]
                predecessors = [last-1, last, last+1]
                option_indices = [j for j in predecessors if 0 <= j and j < w and j not in positions_filled[i]]
                if option_indices == []:
                    k += 1
                    break
                else:
                    option_values = cumulative[i,option_indices]
                    next = option_indices[np.argmin(option_values)]
                    seam_cost += np.min(option_values)
                seam.append(next)
                positions_filled[i].add(next)
            
            else:
                k += 1
                seams.append(seam)
        
        return seams
    # def create_seams(self, cumulative:np.ndarray, num_seams:int, vertically = True) -> list[list]:
    #     if not vertically:
    #         cumulative = cumulative.T
        
    #     h, w = cumulative.shape

    #     seams = []

    #     positions_filled = defaultdict(set)

    #     for k in range(num_seams):

    #         start_index = np.argsort(cumulative[-1,:])[k]
            
    #         seam = [start_index]
    #         positions_filled[h-1].add(start_index)
    #         seam_cost = 0
    #         for i in range(h-1, 0, -1):
                
    #             current = seam[-1]
                
    #             best_option = np.Inf
    #             best_option_index = None

    #             for option in range(current-1, current+2):

    #                 if (option < 0) or (w - 1 < option) or (option in positions_filled[i-1]):
    #                     continue
                
    #                 if cumulative[i-1, option] < best_option:
    #                     best_option = cumulative[i-1, option]
    #                     best_option_index = option
                        
    #             if best_option_index == None:
    #                 best_option_index = current

    #             seam_cost += best_option
    #             positions_filled[i-1].add(best_option_index)
    #             seam.append(best_option_index)            
            
    #         print(seam_cost)
    #         seams.append(seam)
        
    #     return seams
