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
    plt.imshow(np.transpose(image.int().squeeze(0), axes = (1, 2, 0)))
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


def show_sample(sample, idx):
    fig = plt.figure("Batch numéro "+str(idx),figsize=(6, 9))

    images = sample['image']
    images_segm = sample['image_segm']
    for i in range(images.shape[0]):
        image = np.transpose((255*images).int()[i], axes = (1,2,0))
        image_segm = np.transpose((255*images_segm).int()[i], axes = (1,2,0))
        ax = plt.subplot(images.shape[0], 2, 2*i+1)
        ax.set_title('Image originale '+str(i))
        ax.axis('off')
        plt.imshow(image)

        ax = plt.subplot(images.shape[0], 2, 2*i+2)
        ax.set_title('Image segmentée '+str(i))
        ax.axis('off')
        plt.imshow(image_segm)

    plt.tight_layout()
    print("Affichage")
    plt.draw()
    plt.pause(.7)
    plt.close()


def depth(image):
    """
    Returns the 
    """
    return len(torch.from_numpy(image).size())

def min_max_images(segmentation_dataset):
    min_hauteur = 1000
    max_hauteur = 0
    min_largeur = 1000
    max_largeur = 0
    for i in range(len(segmentation_dataset)):
        h = segmentation_dataset[i]['image'].shape[2]
        w = segmentation_dataset[i]['image'].shape[3]
        print(h,w)
        if h < min_hauteur:
            min_hauteur = h
        if h > max_hauteur:
            max_hauteur = h
        if w < min_largeur:
            min_largeur = w
        if w > max_largeur:
            max_largeur = w
        if (i%100==0):
            print(i,"ème image")
    print("Le minimum de taille en hauteur est",min_hauteur,"pixels")
    print("Le maximum de taille en hauteur est",max_hauteur,"pixels")
    print("Le minimum de taille en largeur est",min_largeur,"pixels")
    print("Le maximum de taille en largeur est",max_largeur,"pixels")

def tailles_images(segmentation_dataset):
    max = 640
    min = 145
    tab = [0]*(max-min+1)
    for i in range(len(segmentation_dataset)):
        tab[segmentation_dataset[i]['image'].shape[2]-min] += 1
    total = 0
    for i in range(len(tab)):
        if tab[i] != 0:
            print("Taille",min+i,":",tab[i],"images.")
            total += tab[i]
    print(total,"images dans le tableau")


def calcul_pixel(input_img,outputs,net):
    print("Pixel initial   : ",input_img[0,0,0,0].item(),input_img[0,1,0,0].item(),input_img[0,2,0,0].item())
    print("Pixel final     : ",outputs[0,0,0,0].item(),outputs[0,1,0,0].item(),outputs[0,2,0,0].item())

    R = 0.
    G = 0.
    B = 0.
    u = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            if u == 0:
                R += input_img[0,0,0,0].item()*param[0,0,0,0] + input_img[0,1,0,0].item()*param[0,1,0,0] + input_img[0,2,0,0].item()*param[0,2,0,0]
                G += input_img[0,0,0,0].item()*param[1,0,0,0] + input_img[0,1,0,0].item()*param[1,1,0,0] + input_img[0,2,0,0].item()*param[1,2,0,0]
                B += input_img[0,0,0,0].item()*param[2,0,0,0] + input_img[0,1,0,0].item()*param[2,1,0,0] + input_img[0,2,0,0].item()*param[2,2,0,0]

            if u == 1:
                # print(param)
                R += param[0]
                G += param[1]
                B += param[2]
            u += 1
            # print("net.parameters : ", name, param.data)

    print("Le calcul donne : ",R.item(),G.item(),B.item())


def to_vector(f):
    return f.reshape(-1).unsqueeze(1)


def draw(ylabel,title,batch_size,d,xlim=None,ylim=None):
    plt.xlabel('Number of processed images')
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title+', batch is '+ str(batch_size)+', d = '+str(d))
    plt.grid(True)
    plt.legend()
    plt.tight_layout
