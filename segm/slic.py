# -*- coding: utf-8 -*-
#!/usr/bin/env python

import sys, time, os
from os.path import expanduser

from math import *
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage.segmentation import slic, mark_boundaries
from skimage.color import label2rgb

import metrics
import groundtruth

home = expanduser("~")


# ---------------------------------
# 1: Display segmentation
# ---------------------------------

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
# 2: SLIC segmentation
# ---------------------------------

def slic_segmentation(img, name, nsegments):

    # 1: Process image
    segments = slic(img, n_segments=nsegments, compactness=10.)

    # 3: Compute the metrics
    m = metrics.metrics(segments, name)
    m.set_metrics()

    print("(Eikonal) Boundary recall: " + str(m.recall) + " Density: " +
          str(m.density) + " Undersegmentation: " + str(m.undersegmentationNP) +
          " Precision: " + str(m.precision) + " Compactness: " + str(m.compactness))

    # 3: Output
    return segments, m.recall, m.recall2, m.precision, m.density, m.undersegmentation, m.undersegmentationNP, m.compactness


if __name__ == '__main__':

    img_path = "./segm/data/data9/"
    BSD_path = "./segm/BSD/"
    results_path = "./segm/results/slic9/"
    nsegments = 400

    names, recall, recall2, precision, density, underseg, undersegNP, compactness = [], [], [], [], [], [], [], []
    img_names = os.listdir(home + img_path)

    for idx, name in enumerate(img_names[:]):

        t0 = time.clock()
        print(str(idx) + ": Processing image " + name)
        img = io.imread(home + img_path + name)
        img_BSD = io.imread(home + BSD_path + name[:-4] + '.jpg')
        img_label, r, r2, p, d, u, unp, c = slic_segmentation(img, name, nsegments)
        t1 = time.clock()
        print("time: " + str(t1 - t0))

        display_segmentation(img, img_BSD, img_label, home + results_path + name)

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
    f = open(home + results_path + "result.json", "w")
    f.write(json)
    f.close()
