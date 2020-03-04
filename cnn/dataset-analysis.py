import numpy as np
import random as rd
import matplotlib.pyplot as plt
from collections import Sequence
from itertools import chain, count
import torch
import torchvision
from dataset import *
import torchvision.transforms as transforms


def min_max_images(segmentation_dataset):
    # min_hauteur = 1000
    # max_hauteur = 0
    # min_largeur = 1000
    # max_largeur = 0

    h_vect = []
    w_vect = []
    values = []
    sizes = np.zeros((650, 650), dtype=int)

    for i in range(len(segmentation_dataset)):
        h = segmentation_dataset[i]['input'].shape[0]
        w = segmentation_dataset[i]['input'].shape[1]

        sizes[h][w] += 1

    for h in range(650):
        for w in range(650):
            if sizes[h][w] != 0:
                h_vect.append(h)
                w_vect.append(w)
                values.append(sizes[h][w])

        # if (i % 10 == 0):
        #     print(i, "th image")

    # print("Le minimum de taille en hauteur est", min_hauteur, "pixels")
    # print("Le maximum de taille en hauteur est", max_hauteur, "pixels")
    # print("Le minimum de taille en largeur est", min_largeur, "pixels")
    # print("Le maximum de taille en largeur est", max_largeur, "pixels")

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    plt.figure(figsize=(8, 8))
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)

    ax_scatter.scatter(w_vect, h_vect, s=[10*v for v in values], c=values)

    lim_inf = np.ceil(np.abs([w_vect, h_vect]).min()) - 20
    lim_sup = np.ceil(np.abs([w_vect, h_vect]).max()) + 20
    ax_scatter.set_xlim((lim_inf, lim_sup))
    ax_scatter.set_ylim((lim_inf, lim_sup))

    bins = range(np.abs([w_vect, h_vect]).min(),
                 np.abs([w_vect, h_vect]).max(),
                 10)
    ax_histx.hist(w_vect, bins=bins)
    ax_histy.hist(h_vect, bins=bins, orientation='horizontal')

    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

    plt.show()


if __name__ == "__main__":

    segmentation_dataset = SegmentationDataset(
            root_dir='./data/',
            input_dir='val2017/',
            target_dir='valSP2017/',
            transform=None)

    min_max_images(segmentation_dataset)
