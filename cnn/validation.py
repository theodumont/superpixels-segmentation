"""
Validation methods for the neural network
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
import numpy as np

import sys, time, os


def validation(model_idx, epoch_idx):

    # Select a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # Training parameters
    batch_size = 2

    # Load the segmentation dataset
    segmentation_dataset = SegmentationDataset(
        root_dir='./data/',
        input_dir='val2017/',
        target_dir='valSP2017/',
        transform=transforms.Compose([
             RandomCrop(224),
             Normalize(),
             ToTensor()]))

    # Data loader
    val_loader = torch.utils.data.DataLoader(
        segmentation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0)

    # Initializes the neural network
    net = Net(8)
    net.to(device)
    PATH = './results/weights/run'+str(model_idx)+'_'+str(epoch_idx)+'.pth'
    net.load_state_dict(torch.load(PATH))

    # Compute loss
    loss_epoch = []
    criterion = lambda outputs, target: TV_loss(outputs, target,
                                                batch_size=batch_size,
                                                alpha=0.)

    running_loss = 0.0
    for i, sample in enumerate(val_loader):

        # Load batch
        input_img = sample['input'].to(device)
        segm_true = sample['target'].to(device)

        # Forward/Backward pass
        outputs = net(input_img)
        loss = criterion(outputs, segm_true)
        loss_epoch.append(loss.item())
        running_loss += loss.item()
        if(i % 20 == 0):
            print('[%d,%d] loss: %.6f' % (1, i+1, running_loss/(i+1)))

    return np.array(loss_epoch)


if __name__ == '__main__':

    losses = np.zeros(30)
    model_idx = 12
    for idx in range(30):
        loss = validation(model_idx, idx)
        print("Epoch: " + str(idx + 1) + " Loss: " + str(np.mean(loss)))
        losses[idx] = np.mean(loss)

    np.save("./results/loss-validation/Validation_loss_"+str(model_idx)+".npy", losses)
