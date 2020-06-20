"""
Useful tools and functions
"""

import random as rd
import numpy as np
import matplotlib.pyplot as plt
import torch
from itertools import chain, count
from collections import Sequence


def show_image(image):

    """
    Display an image

    :param image: Image to be displayed
    :type image: Tensor
    """

    plt.figure(0)
    plt.imshow(np.transpose(image.int().squeeze(0), axes=(1, 2, 0)))
    plt.draw()
    plt.pause(.7)
    plt.close()


def show_sample_index(segmentation_dataset, idx_list):

    """
    Display a sample image from the dataset

    :param segmentation_dataset: Segmentation dataset
    :type segmentation_dataset: Tensor
    """

    for idx in idx_list:
        sample = segmentation_dataset[idx]
        show_sample(sample, idx)


def show_sample(sample):
    fig = plt.figure(0)
    ax = plt.subplot(112)
    ax.imshow(np.transpose(sample['input'].int().squeeze(0), axes=(1, 2, 0)))
    ax = plt.subplot(212)
    ax.imshow(np.transpose(sample['target'].int().squeeze(0), axes=(1, 2, 0)))
    plt.draw()
    plt.pause(.7)
    plt.close()


def show_batch(batch):
    """
    Display a sample image from the dataset, v2
    """
    fig = plt.figure("Batch".format(idx))

    images = batch['image']
    images_segm = batch['image_segm']
    for i in range(images.shape[0]):
        image = np.transpose((255*images).int()[i], axes=(1, 2, 0))
        image_segm = np.transpose((255*images_segm).int()[i], axes=(1, 2, 0))
        ax = plt.subplot(images.shape[0], 2, 2*i+1)
        ax.set_title('Original image {}'.format(i))
        ax.axis('off')
        plt.imshow(image)

        ax = plt.subplot(images.shape[0], 2, 2*i+2)
        ax.set_title('Segmented image {}'.format(i))
        ax.axis('off')
        plt.imshow(image_segm)

    plt.tight_layout()
    print("Display...")
    plt.draw()
    plt.pause(.7)
    plt.close()


def depth(image):
    """
    Return the depth of an image
    """
    return len(torch.from_numpy(image).size())


def to_vector(f):
    """
    Transform a tensor into a vector
    """
    return f.reshape(-1).unsqueeze(1)


def draw(ylabel, title, batch_size, d, xlim=None, ylim=None):
    plt.xlabel('Number of processed images')
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title('{}, batch is {}, d = {}'.format(title, batch_size, d))
    plt.grid(True)
    plt.legend()
    plt.tight_layout
