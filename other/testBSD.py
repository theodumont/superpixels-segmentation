"""
Training methods for the neural network
"""

import sys, time, os
from os.path import expanduser
home = expanduser("~")

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
from skimage.util import img_as_ubyte
from skimage.io import imsave


def rescale(output):

    """
    Rescale the output of the network
    """
    out = output.cpu().detach().numpy()[0]
    out = np.transpose(out.T, axes = (1, 0, 2))
    out -= np.min(out)
    out /= np.max(out)
    return img_as_ubyte(out)


if __name__ == '__main__':

    display = False
    output_path = './data13/'

    # Select a device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ",device)

    # Training parameters
    batch_size = 1

    # Load the segmentation dataset
    segmentation_dataset = SegmentationDataset(
        root_dir= '../../Segmentation/data/Berkeley/',
        input_dir= 'test/',
        target_dir= 'test/',
        transform=transforms.Compose([
             Normalize(),
             ToTensor()]))

    # Data loader
    testloader = torch.utils.data.DataLoader(
        segmentation_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 0)

    # Initializes the neural network
    net = Net(7)
    net.to(device)
    PATH = './weights/weights_13.pth'
    net.load_state_dict(torch.load(PATH))

    for i, sample in enumerate(testloader):

        input_img = sample['input'].to(device)
        name = sample['name'][0]
        print(name)
        output_img = net(input_img)

        # Display result
        imIn = rescale(input_img)
        imResult = rescale(output_img)

        if(display):
            fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True, 
              sharey=True, subplot_kw={'adjustable': 'box-forced'})
            ax[0].imshow(imIn)
            ax[1].imshow(imResult)

            for a in ax.ravel():
                a.set_axis_off()
            plt.tight_layout()
            plt.show()

        # Save image
        imsave(output_path + name + '.png', imResult)
        



