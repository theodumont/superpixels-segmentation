"""
Useful tools and functions
"""
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from collections import Sequence
from itertools import chain, count
import torch
import torchvision


def show_image(image):

    """
    Display an image

    :param image: Image to be displayed
    :type image: Tensor
    """

    plt.figure(0)
    plt.imshow(np.transpose(image.int().squeeze(0), axes=(1, 2, 0)))
    plt.draw()
    plt.pause(5)
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


def show_sample(sample, idx):
    fig = plt.figure("Batch number "+str(idx), figsize=(6, 9))

    images = sample['image']
    images_segm = sample['image_segm']
    for i in range(images.shape[0]):
        image = np.transpose((255*images).int()[i], axes=(1, 2, 0))
        image_segm = np.transpose((255*images_segm).int()[i], axes=(1, 2, 0))
        ax = plt.subplot(images.shape[0], 2, 2*i+1)
        ax.set_title('Original image '+str(i))
        ax.axis('off')
        plt.imshow(image)

        ax = plt.subplot(images.shape[0], 2, 2*i+2)
        ax.set_title('Segmented image '+str(i))
        ax.axis('off')
        plt.imshow(image_segm)

    plt.tight_layout()
    print("Display...")
    plt.draw()
    plt.pause(.7)
    plt.close()


def depth(image):

    """
    Returns the size of an image
    """

    return len(torch.from_numpy(image).size())


def min_max_images(segmentation_dataset):

    """
    Returns the minima and maxima of the dataset images' size.

    :param segmentation_dataset: Segmentation dataset
    :type segmentation_dataset: Tensor
    """

    min_heigth = 1000
    max_heigth = 0
    min_width = 1000
    max_width = 0
    for i in range(len(segmentation_dataset)):
        h = segmentation_dataset[i]['image'].shape[2]
        w = segmentation_dataset[i]['image'].shape[3]
        print(h, w)
        if h < min_heigth:
            min_heigth = h
        if h > max_heigth:
            max_heigth = h
        if w < min_width:
            min_width = w
        if w > max_width:
            max_width = w
        if (i % 100 == 0):
            print(i, "th image")
    print("Minimum height is", min_heigth, "pixels")
    print("Maximum height is", max_heigth, "pixels")
    print("Minimum width is", min_width, "pixels")
    print("Maximum width is", max_width, "pixels")


def calcul_pixel(input_img, outputs, net):

    """
    Tests if the convolution computation of a network is correct.

    :param input_img: Image we want to test
    :param outputs: Image segmented by the network
    :param net: Network

    :type input_img: Tensor
    :type outputs: Tensor
    :type net: Net
    """

    print("Initial pixel   : ", input_img[0, 0, 0, 0].item(), input_img[0, 1, 0, 0].item(), input_img[0, 2, 0, 0].item())
    print("Final pixel     : ", outputs[0, 0, 0, 0].item(), outputs[0, 1, 0, 0].item(), outputs[0, 2, 0, 0].item())

    R = 0.
    G = 0.
    B = 0.
    u = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            if u == 0:
                R += input_img[0, 0, 0, 0].item()*param[0, 0, 0, 0] + input_img[0, 1, 0, 0].item()*param[0, 1, 0, 0] + input_img[0, 2, 0, 0].item()*param[0, 2, 0, 0]
                G += input_img[0, 0, 0, 0].item()*param[1, 0, 0, 0] + input_img[0, 1, 0, 0].item()*param[1, 1, 0, 0] + input_img[0, 2, 0, 0].item()*param[1, 2, 0, 0]
                B += input_img[0, 0, 0, 0].item()*param[2, 0, 0, 0] + input_img[0, 1, 0, 0].item()*param[2, 1, 0, 0] + input_img[0, 2, 0, 0].item()*param[2, 2, 0, 0]

            if u == 1:
                R += param[0]
                G += param[1]
                B += param[2]
            u += 1

    print("Par calcul : ", R.item(), G.item(), B.item())


def to_vector(f):
    return f.reshape(-1).unsqueeze(1)


def draw(ylabel, title, batch_size, d, xlim=None, ylim=None):
    plt.xlabel('Number of processed images')
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title+', batch is ' + str(batch_size)+', d = '+str(d))
    plt.grid(True)
    plt.legend()
    plt.tight_layout
