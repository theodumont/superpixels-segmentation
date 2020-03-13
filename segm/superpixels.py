# -*- coding: utf-8 -*-
#!/usr/bin/env python

import sys, time, os
from os.path import expanduser

from math import *
import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage.segmentation import slic

import metrics
import eikonal
import groundtruth

home = expanduser("~")


# ---------------------------------
# 1: Eikonal segmentation
# ---------------------------------

def eikonal_segmentation(img_path, results_path, name, nsegments, width, weights):

    # 1: Define the parameters
    update = False

    # 2: Process image
    img = io.imread(home + img_path + name)
    inst = eikonal.eikonal_superpixels(img, name, update)
    inst.set_seeds(width)
    inst.segmentation_groundtruth(weights)
    nsuperpixels = len(inst.seeds)//2

    while(nsuperpixels < nsegments):
        n = min(20, nsegments - nsuperpixels)
        inst.refine_segmentation(n)
        nsuperpixels += n

    # 3: Compute the metrics
    m = metrics.metrics(inst.lb - 1, name)
    m.set_metrics()

    print("(Eikonal) Boundary recall: " + str(m.recall) + " Recall: " +
          str(m.recall2) + " Undersegmentation: " + str(m.undersegmentation) +
          " Precision: " + str(m.precision) + " Compactness: " + str(m.compactness))

    # 4: Output
    return inst, m.recall, m.recall2, m.precision, m.density, m.undersegmentation, m.undersegmentationNP, m.compactness


# ---------------------------------
# 2: SLIC segmentation
# ---------------------------------

def slic_segmentation(img_path, name, nsegments):

    # 1: Process image
    img = io.imread(home + img_path + name)
    segments = slic(img, n_segments=nsegments, compactness=1)

    # 3: Compute the metrics
    m = metrics.superpixel_metrics(segments, name)
    m.set_metrics()

    print("(Eikonal) Boundary recall: " + str(m.recall) + " Density: " +
          str(m.density) + " Undersegmentation: " + str(m.undersegmentationNP) +
          " Precision: " + str(m.precision) + " Compactness: " + str(m.compactness))

    # 3: Output
    return m.recall, m.recall2, m.precision, m.density, m.undersegmentation, m.undersegmentationNP, m.compactness


if __name__ == '__main__':

    img_path = "./segm/data/data10/"
    BSD_path = "./segm/BSD/"
    results_path = "./segm/results/run10/"

    width = 25
    nsegments = 400
    weights = [7., 0., 0.]

    names, recall, recall2, precision, density, underseg, undersegNP, compactness = [], [], [], [], [], [], [], []
    img_names = os.listdir(home + img_path)

    for idx, name in enumerate(img_names[:]):

        t0 = time.clock()
        print(str(idx) + ": Processing image " + name)
        inst, r, r2, p, d, u, unp, c = eikonal_segmentation(img_path, results_path, name, nsegments, width, weights)
        t1 = time.clock()
        print("time: " + str(t1 - t0))

        img_BSD = io.imread(home + BSD_path + name[:-4] + '.jpg')
        inst.display_segmentation(img_BSD, home + results_path + name)

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
