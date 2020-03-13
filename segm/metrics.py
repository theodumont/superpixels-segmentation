# -*- coding: utf-8 -*-
#!/usr/bin/env python

import sys, time, os
from os.path import expanduser

from math import *
import numpy as np
import groundtruth

from skimage import io, color
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, rectangle, disk
from skimage.measure import regionprops
import matplotlib.pyplot as plt

home = expanduser("~")


class metrics:

    """
    Compute the metrics associated to a segmentation for the
    Berkeley Segmentation Dataset
    """

    def __init__(self, lb, name):

        """
        Class constructor

        :param lb: Label image corresponding to the segmentation
        :param name: Image name
        """

        self.lb = lb.astype('int')
        if(np.min(self.lb) > 0):
            self.lb -= 1

        self.nx, self.ny = self.lb.shape
        self.img_truth, self.segments_truth = groundtruth.get_segment_from_filename(name[:-4])

    # -------------------------------------------
    # Boundary recall and precision
    # -------------------------------------------

    def set_boundary_recall(self, s1=5, s2=3):

        """
        Calculate the boundary recall with respect to the ground truth
        :param s1: Size of the first rectangle used to dilate the boundary
        (default=5)
        :param s2: Size of the second rectangle used to dilate the boundary
        (default=3)

        """

        # s1
        eikonal_bd = dilation(find_boundaries(self.lb), rectangle(s1, s1))
        self.recall = 0
        for idx, truth in enumerate(self.img_truth):
            self.recall += float(np.sum(eikonal_bd * truth)) / float(np.sum(truth))
        self.recall /= len(self.img_truth)

        # s2
        eikonal_bd = dilation(find_boundaries(self.lb), rectangle(s2, s2))
        self.recall2 = 0
        for idx, truth in enumerate(self.img_truth):
            self.recall2 += float(np.sum(eikonal_bd * truth)) / float(np.sum(truth))
        self.recall2 /= len(self.img_truth)

    def set_boundary_precision(self, s=5):

        """
        Compute the boundary precision with respect to the ground truth
        :param s: Size of the rectangle used to dilate the boundary (default=5)
        """

        # s=5
        eikonal_bd = find_boundaries(self.lb)
        self.precision = 0
        global_score = float(np.sum(eikonal_bd))

        for idx, truth in enumerate(self.img_truth):
            intersection = eikonal_bd * dilation(truth, rectangle(s, s))
            self.precision += float(np.sum(intersection)) / global_score

        self.precision /= len(self.img_truth)

    # -------------------------------------------
    # Undersegmentation
    # -------------------------------------------

    def set_undersegmentation(self):

        """
        Compute the undersegmentation metric
        """

        n_segments = int(np.max(self.lb)) + 1
        self.undersegmentation = 0.

        for truth in self.segments_truth:

            n_labels = int(np.max(truth) + 1)
            area = np.zeros(n_segments)
            hist = np.zeros((n_segments, n_labels))
            u = 0

            for x in range(self.nx):
                for y in range(self.ny):

                    idx = int(self.lb[x, y])
                    t_idx = int(truth[x, y])
                    hist[idx, t_idx] += 1
                    area[idx] += 1

            for k in range(n_segments):
                u += area[k] - np.max(hist[k, :])

            u /= self.nx*self.ny
            self.undersegmentation += u

        self.undersegmentation /= len(self.segments_truth)

    def set_undersegmentationNP(self):
        """
        """
        self.undersegmentationNP = 0

        for truth in self.segments_truth:
            u = 0
            inter_mat, s_sizes = self.computeInsersectionMatrix(truth)
            num_g, num_s = inter_mat.shape
            for i in range(num_g):
                for j in range(num_s):
                    u += min(inter_mat[i, j], s_sizes[j] - inter_mat[i, j])

            u /= float(self.nx * self.ny)
            self.undersegmentationNP += u

        self.undersegmentationNP /= len(self.segments_truth)

    def computeInsersectionMatrix(self, truth):
        """
        """
        num_s = int(self.lb.max()) + 1
        num_g = int(truth.max()) + 1
        inter_mat = np.zeros((num_g, num_s), dtype='int64')
        superpixel_sizes = np.zeros((num_s,), dtype='int64')

        for i in range(self.nx):
            for j in range(self.ny):
                inter_mat[int(truth[i, j]), int(self.lb[i, j])] += 1
                superpixel_sizes[int(self.lb[i, j])] += 1

        return inter_mat, superpixel_sizes

    # -------------------------------------------
    # Geometrical criteria
    # -------------------------------------------

    def set_density(self):

        """
        Computes the segmentation density
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
                if(i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1):
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
            ratio = area/float(max_area)
            if(perimeter > 0):
                self.compactness += 4 * pi * ratio * area / pow(perimeter, 2)

    def set_metrics(self):

        """
        Computes all metrics
        """

        self.set_boundary_recall()
        self.set_boundary_precision()
        self.set_density()
        self.set_undersegmentation()
        self.set_undersegmentationNP()
        self.set_compactness()


# ---------------------------------
# When executed, run the main script
# ---------------------------------

if __name__ == '__main__':

    # Test undersegmentation

    gt = np.zeros((10, 10))
    gt[:, :5] = np.ones((10, 5))

    seg = np.zeros((10, 10))
    seg[:, :2] = np.ones((10, 2))
    seg[5:, :5] = np.ones((5, 5))
    seg[:5, 2:] = 2*np.ones((5, 8))

    n_segments = 2
    n_labels = 3
    area = np.zeros(n_labels)
    hist = np.zeros((n_labels, n_segments))

    for x in range(10):
        for y in range(10):

            idx = int(seg[x, y])
            t_idx = int(gt[x, y])
            hist[idx, t_idx] += 1
            area[idx] += 1

    u = 0
    for k in range(n_labels):
        u += area[k] - np.max(hist[k, :])
