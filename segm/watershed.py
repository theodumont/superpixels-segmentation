# -*- coding: utf-8 -*-
#!/usr/bin/env python

import sys, time, os
from os.path import expanduser

from math import *
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage.segmentation import slic
from skimage.color import rgb2gray
from skimage.segmentation import watershed, mark_boundaries
from skimage.filters import sobel
from skimage.color import label2rgb

import metrics
import groundtruth

home = expanduser("~")


def display_segmentation(img, img_BSD, img_label, output_path=None):

    """
    Display the result of the superpixel segmentation (Watershed)
    """

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    ax[0, 0].imshow(img_BSD)
    ax[0, 1].imshow(img)
    ax[1, 0].imshow(mark_boundaries(img_BSD, img_label))
    ax[1, 1].imshow(label2rgb(img_label, img_BSD, kind='avg'))

    for a in ax.ravel():
        a.set_axis_off()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


# ---------------------------------
# 1: Eikonal segmentation
# ---------------------------------

def watershed_segmentation(img_path, BSD_path, results_path, name, nsegments):

    # 1: Define the parameters
    update = False

    # 2: Process image
    img_BSD = io.imread(BSD_path + name[:-4] + ".jpg")
    img = io.imread(img_path + name)
    img_gray = rgb2gray(img)
    gradient = sobel(rgb2gray(img))
    img_label = watershed(gradient, nsegments)

    display_segmentation(img, img_BSD, img_label, results_path + name[:-4] + ".png")

    # 3: Compute the metrics
    m = metrics.metrics(img_label - 1, name)
    m.set_metrics()

    print("(Eikonal) Boundary recall: " + str(m.recall) + " Recall: " +
          str(m.recall2) + " Undersegmentation: " + str(m.undersegmentation) +
          " Precision: " + str(m.precision) + " Compactness: " + str(m.compactness))

    # 3: Output
    return m.recall, m.recall2, m.precision, m.density, m.undersegmentation, m.undersegmentationNP, m.compactness


if __name__ == '__main__':

    img_path = home + "./segm/data/data9/"
    BSD_path = home + "./segm/BSD/"
    results_path = home + "./segm/results/ws9/"

    width = 25
    nsegments = 500
    weights = [7., 0., 0.]

    names, recall, recall2, precision, density, underseg, undersegNP, compactness = [], [], [], [], [], [], [], []
    img_names = os.listdir(img_path)

    for idx, name in enumerate(img_names[:2]):

        t0 = time.clock()
        print(str(idx) + ": Processing image " + name)
        r, r2, p, d, u, unp, c = watershed_segmentation(img_path, BSD_path, results_path, name, nsegments)
        t1 = time.clock()
        print("time: " + str(t1 - t0))

        names.append(name)
        recall.append(r)
        recall2.append(r2)
        precision.append(p)
        density.append(d)
        underseg.append(u)
        undersegNP.append(unp)
        compactness.append(c)

    results = {'name': names, 'recall': recall, 'recall2': recall2,
               'precision': precision, 'density': density,
               'undersegmentation': underseg, 'undersegmentationNP': undersegNP,
               'compactness': compactness}

    import json
    json = json.dumps(results)
    f = open(results_path + "result.json", "w")
    f.write(json)
    f.close()
