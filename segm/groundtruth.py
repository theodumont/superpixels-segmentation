# -*- coding: utf-8 -*-
#!/usr/bin/env python

import sys, time, os
from os.path import expanduser

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import find_boundaries

home = expanduser("~")


def get_segmentation(path, filename):

    """
    Load groundtruth on BSD500 for the specified image
    """

    f = loadmat(path + filename)
    data = f['groundTruth'][0]
    groundtruth = []
    for img in data:
        groundtruth.append(img[0][0][0])

    return groundtruth


def get_segment_from_filename(filename):

    """
    Load groundtruth on BSD500 for the specified image
    """

    path = home + "/Superpixels/data/Truth/"
    list_dir = os.listdir(path)
    filename += '.mat'

    segments = []
    for folder in list_dir:

        list_img = os.listdir(path + folder + "/")
        if(filename in list_img):
            segments.extend(get_segmentation(path + folder + "/", filename))

    boundaries = []
    for segment in segments:
        boundaries.append(find_boundaries(segment))

    return boundaries, segments


# ---------------------------------
# When executed, run the main script
# ---------------------------------

if __name__ == '__main__':

    filename = "2092"
    boundaries, segments = get_segment_from_filename(filename)
    plt.imshow(boundaries[0])
    plt.show()
