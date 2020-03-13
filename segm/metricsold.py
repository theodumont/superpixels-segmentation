# -*- coding: utf-8 -*-
#!/usr/bin/env python

import sys, time, os
from os.path import expanduser

from math import *
import numpy as np

from skimage import io, color
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, rectangle, disk
import matplotlib.pyplot as plt

home = expanduser("~")


class superpixel_metrics:

    """
    Compute the metrics associated to a given superpixel segmentation
    """

    def __init__(self, lb):
        """
        Class constructor

        :param lb: Label image containing the superpixel partition of the image
        (2D Numpy array). Corresponds for instance to the label image outputed
        by the SLIC implementation in scikit-image.
        """

        self.lb = lb
        self.nx, self.ny = self.lb.shape

        # TO DO
        # Load the image containing the ground truth contours as self.img_truth

    def set_boundary_recall(self, s=5):
        """
        Calculate the boundary recall with respect to the ground truth
        :param s: Size of the rectangle used to dilate the boundary (default=5)
        """

        eikonal_bd = dilation(find_boundaries(self.lb), rectangle(s, s))
        self.recall = float(np.sum(eikonal_bd * truth)) / float(np.sum(self.img_truth))

    def set_boundary_precision(self, s=5):
        """
        Compute the boundary precision with respect to the ground truth
        :param s: Size of the rectangle used to dilate the boundary (default=5)
        """

        eikonal_bd = find_boundaries(self.lb)
        intersection = eikonal_bd * dilation(truth, rectangle(s, s))
        self.precision += float(np.sum(intersection)) / float(np.sum(eikonal_bd))

    def set_undersegmentation(self):
        """
        Undersegmentation
        """

        self.undersegmentation = 0.

        n_segments = int(np.max(self.lb)) + 1    # Number of superpixels
        n_labels = int(np.max(truth) + 1)        # Number of regions in the ground truth
        area = np.zeros(n_segments)              # Areas of the superpixels
        hist = np.zeros((n_segments, n_labels))  # Overlap with the ground truth

        # Process the image pixel by pixel in a single pass
        for x in range(self.nx):
            for y in range(self.ny):

                # label of the superpixel covering the current pixel
                idx = int(self.lb[x, y])

                # Label of the corresponding ground truth
                t_idx = int(truth[x, y])

                hist[idx, t_idx] += 1
                area[idx] += 1

        # Computes the superpixel leakage
        for k in range(n_segments):
            self.undersegmentation += area[k] - np.max(hist[k, :])

        self.undersegmentation /= self.nx * self.ny

    def set_density(self):
        """
        Measures the superpixels density
        """

        self.density = np.sum(find_boundaries(self.lb)) / float(self.nx * self.ny)

    def perimeter(self):
        """
        Computes the perimeter of the superpixels
        """

        num = self.lb.max() + 1
        self.perimeters = np.zeros((num))

        for i in range(self.nx):
            for j in range(self.ny):

                label = self.lb[i, j]

                # Pixel belonging to an image border
                if(i == 0 or i == self.nx - 1 or
                   j == 0 or j == self.ny - 1):
                    self.perimeters[label] += 1

                # Pixel at the border of two regions
                elif(self.lb[i - 1, j] != label or self.lb[i + 1, j] != label or
                     self.lb[i, j - 1] != label or self.lb[i, j + 1] != label):
                    self.perimeters[label] += 1

    def set_compactness(self):
        """
        Compute compactness
        """

        # Compute the segments perimeters
        self.perimeter()

        # Compute compactness
        self.compactness = 0
        max_area = self.nx * self.ny

        for i in range(self.lb.max() + 1):

            idx = np.where(self.lb == i)
            area = len(idx[0])
            perimeter = self.perimeters[i]
            ratio = area/max_area
            if(perimeter > 0):
                self.compactness += 4*pi*ratio*area/pow(perimeter, 2)
