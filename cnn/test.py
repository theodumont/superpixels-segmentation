"""
Training methods for the neural network
"""

import torch
import torchvision
import torchvision.transforms as transforms
from dataset import *
from utils import *
import torch.nn as nn
import torch.nn.functional as F
from loss import TV_loss
import matplotlib.pyplot as plt
from network import AdaptiveBatchNorm2d, ChenConv
from network import Net

from skimage.morphology import disk
from skimage.filters.rank import gradient

import numpy as np

import sys, time, os


if __name__ == '__main__':

    # Select a device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # Training parameters
    batch_size = 2

    # Load the segmentation dataset
    segmentation_dataset = SegmentationDataset(
        root_dir='../../../../data/commun/COCO/',
        input_dir='val2017/',
        target_dir='valSP2017/',
        transform=transforms.Compose([
             RandomCrop(224),
             Normalize(),
             ToTensor()]))

    # Data loader
    testloader = torch.utils.data.DataLoader(
        segmentation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)

    # Initializes the neural network
    net = Net(6)
    net.to(device)
    PATH = './../results/weights/run5_14.pth'
    net.load_state_dict(torch.load(PATH))

    for i, sample in enumerate(testloader):

        input_img = sample['input'].to(device)
        segm_true = sample['target'].to(device)

        outputs = net(input_img)
        if(i > 1):
            break

    # Display result
    imIn = input_img.cpu().detach().numpy()[0]
    imTarget = segm_true.cpu().detach().numpy()[0]
    imResult = outputs.cpu().detach().numpy()[0]
    fig, ax = plt.subplots(2, 2, figsize=(10, 10),
                           sharex=True, sharey=True,
                           subplot_kw={'adjustable': 'box-forced'})
    ax[0, 0].imshow(np.transpose(imIn.T, axes=(1, 0, 2)))
    ax[0, 1].imshow(np.transpose(imResult.T, axes=(1, 0, 2)))
    ax[1, 0].imshow(np.transpose(imTarget.T, axes=(1, 0, 2)))
    ax[1, 1].imshow(np.transpose(imIn.T, axes=(1, 0, 2)))

    for a in ax.ravel():
        a.set_axis_off()
    plt.tight_layout()
    plt.show()
