# -*- coding: utf-8 -*-
#!/usr/bin/env python

import sys, time, os
from os.path import expanduser

from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.signal import argrelextrema
from scipy.ndimage.filters import gaussian_filter

import libfmm as fmm
import groundtruth
import metrics

from skimage import io, color
from skimage.color import rgb2lab, rgb2gray, label2rgb
from skimage.segmentation import mark_boundaries, find_boundaries, slic
from skimage.filters import gabor_kernel
from skimage.morphology import extrema, disk, dilation, rectangle
from skimage.measure import regionprops, label
from skimage.filters.rank import gradient

home = expanduser("~")
sys.path.append("./src/lib/Release/")


# ---------------------------
# 1: Eikonal based superpixels
# ---------------------------

class eikonal_superpixels:

    """
    Superpixel segmentation based upon the resolution of Eikonal equation.
    """

    def __init__(self, img, name, update=False):

        """
        Class constructor
        """

        # Open an image and convert it into a LAB image
        self.img = img[:, :]
        self.img_lab = rgb2lab(self.img)
        self.img_gray = rgb2gray(self.img)
        self.img_grad = gradient(self.img_gray, disk(1))
        self.nx, self.ny, self.nchannels = self.img_lab.shape
        self.update = update

        # LAB channels
        for k in range(3):
            m = np.mean(self.img_lab[:, :, k])
            st = np.std(self.img_lab[:, :, k])
            self.img_lab[:, :, k] = (self.img_lab[:, :, k] - m)/st
        self.im_lab = list(self.img_lab.ravel())
        self.nchannels = 3

        # Obtain the corresponding groundtruth images
        self.img_truth, self.segments_truth = groundtruth.get_segment_from_filename(name[:-4])

    # ---------------------------------
    # Initial Segmentation
    # ---------------------------------

    def set_seeds(self, width):

        """
        Set the initial germs at local minima of the gradient
        """

        self.seeds = []
        nx = self.nx // width
        ny = self.ny // width

        for p in range(nx):
            for q in range(ny):

                dx, dy = width//2, width//2
                sx, sy = width//4, width//4

                if(p == nx - 1):
                    sx = 0
                    dx = self.nx - p*width
                if(q == ny - 1):
                    sy = 0
                    dy = self.ny - q*width

                arr = self.img_grad[width*p + sx:width*p + sx + dx,
                                    width*q + sy:width*q + sy + dy]

                m = np.argmin(arr)
                x = m // dy
                y = m % dy

                self.seeds.extend([width*p + sx + x, width*q + sy + y])

    def segmentation(self, weights):

        """
        Superpixel generation
        """

        self.eikonal = fmm.Fast_marching(self.nx, self.ny, 0, self.im_lab, self.im_lab, self.seeds, weights, self.update)
        self.eikonal.run()
        lb, dist, contours = self.eikonal.get_results()
        self.lb = np.array(lb).reshape((self.nx, self.ny))
        self.dist = np.array(dist).reshape((self.nx, self.ny))
        self.contours = np.array(contours).reshape((self.nx, self.ny))

    def segmentation_groundtruth(self, weights):

        """
        Superpixel generation consistent with the ground truth
        """

        ground_truth = list(np.array(self.segments_truth[0].astype('double')).ravel())
        self.eikonal = fmm.Fast_marching(self.nx, self.ny, 1, self.im_lab, ground_truth, self.seeds, weights, self.update)
        self.eikonal.run()
        lb, dist, contours = self.eikonal.get_results()
        self.lb = np.array(lb).reshape((self.nx, self.ny))
        self.dist = np.array(dist).reshape((self.nx, self.ny))
        self.contours = np.array(contours).reshape((self.nx, self.ny))

    # ---------------------------------
    # Initial Segmentation
    # ---------------------------------

    def add_seeds(self, N):

        """
        Extract the region properties
        """
        boundaries = dilation(find_boundaries(self.lb), disk(3))
        regions = regionprops(self.lb, self.dist - boundaries*self.dist)

        n_seeds = []
        max_dist = []
        for region in regions:

            w, h = region.intensity_image.shape
            m = np.argmax(region.intensity_image)
            x = m//h + region.bbox[0]
            y = m % h + region.bbox[1]
            n_seeds.append([x, y])
            max_dist.append(np.max(region.intensity_image))

        arg_idx = np.argsort(np.array(max_dist))[::-1]

        for idx in arg_idx[:N]:
            self.seeds.extend(n_seeds[idx])

    def refine_segmentation(self, N):

        """
        Add new superpixels from the current segmentation
        """

        self.add_seeds(N)
        self.eikonal.add_seeds(self.seeds)
        self.eikonal.run()
        lb, dist, contours = self.eikonal.get_results()
        self.lb = np.array(lb).reshape((self.nx, self.ny))
        self.dist = np.array(dist).reshape((self.nx, self.ny))
        self.contours = np.array(contours).reshape((self.nx, self.ny))

    # ---------------------------------
    # Display
    # ---------------------------------

    def save(self, filename):

        """
        Save the label image as a numpy file

        :param filename: Name of the file containing the labels
        :type filename: String
        """
        np.save(filename, self.lb - 1)

    def display_segmentation(self, img_BSD, output_path=None):

        """
        Display the result of the superpixel segmentation
        """
        fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
        ax[0, 0].imshow(img_BSD)
        ax[0, 1].imshow(self.img)
        ax[1, 0].imshow(mark_boundaries(img_BSD, self.lb))
        # ax[1, 0].imshow(img_BSD - self.img)
        ax[1, 1].imshow(label2rgb(self.lb, img_BSD, kind='avg'))

        for a in ax.ravel():
            a.set_axis_off()
        plt.tight_layout()

        if output_path is not None:
            plt.savefig(output_path)
        else:
            plt.show()

    def display_ground_truth(self):

        """
        Display the ground truth segmentation
        """

        fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
        ax[0].imshow(self.img_truth[0])
        ax[1].imshow(self.segments_truth[0])
        for a in ax.ravel():
            a.set_axis_off()
        plt.tight_layout()
        plt.show()

    def display_seeds(self):

        """
        Display the seeds
        """

        plt.figure()
        plt.imshow(self.img)
        plt.scatter(np.array(self.seeds[1::2]), np.array(self.seeds[0::2]))
        plt.show()

    def display_texture(self):

        """
        Display the texture images
        """
        fig, ax = plt.subplots(4, 2, figsize=(10, 10), sharex=True, sharey=True)

        for q in range(4):
            for s in range(2):
                ax[q, s].imshow(self.img_texture[:, :, 2*q + s])

        for a in ax.ravel():
            a.set_axis_off()
        plt.tight_layout()
        plt.show()
