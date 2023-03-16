import numpy as np
from PIL import Image
from tqdm import tqdm

from gradientmagniture import GradientMagnitude


class SeamCarver:

    def __init__(self, image: Image, grad_func: GradientMagnitude) -> None:
        self.im = np.asarray(image)
        self.grad_func = grad_func
        
        if len(self.im.shape) < 3:
            self.im = self.im[..., np.newaxis] # get the image on the shape (x,y,1)
    
    def _compute_energy_map(self, im):
        """ Compute the energy using the gradient function """
        h,w,d = im.shape

        energy_map = np.zeros((h, w))
        for i in range(d):
            energy_map += self.grad_func.grad(im[:,:,i])
        return energy_map

    def _compute_cost_map(self, energy_map):
        """ Compute the cumulative cost moving top-down in the image """
        cost = np.copy(energy_map)
        h,w = cost.shape

        for i in range(1, h):
            for j in range(0, w):
                offset = 0 if j == 0 else 1 #TODO Maybe there is a more elegant way of doing this
                cost[i,j] += np.min(cost[i-1,j-offset:j+2])
        return cost

    def _compute_min_energy_seam(self, cost_map):
        """ Returns a mask of the same size as the cost-map containing the locations of the seam """
        seam_mask = np.ones_like(cost_map, dtype=bool)
        h,_ = cost_map.shape

        # Find starting point
        last_index = np.argmin(cost_map[-1,:])
        seam_mask[h-1, last_index] = False

        # Backtrack
        for i in range(2, h+1):
            offset = 0 if last_index == 0 else 1 #TODO Maybe there is a more elegant way of doing this
            last_index = np.argmin(cost_map[h-i+1, last_index-offset : last_index+2]) + last_index - offset
            seam_mask[h-i, last_index] = False
        return seam_mask
    
    def _remove_seam(self, im, seam_mask, n_seams=1):
        """ Remove a seam based on a mask of the same size as the image """
        h,w,d = im.shape
        return im[seam_mask].reshape(h, w-n_seams,d)


    def carve(self, n_seams = 20):
        """ Remove n seams from the image """
        new_im = np.copy(self.im)
        
        for _ in tqdm(range(n_seams)):
            
            energy_map = self._compute_energy_map(new_im)
            cost_map = self._compute_cost_map(energy_map)
            min_energy_seam = self._compute_min_energy_seam(cost_map)
            new_im = self._remove_seam(new_im, min_energy_seam)

        return new_im
