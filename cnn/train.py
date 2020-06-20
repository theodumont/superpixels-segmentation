"""
Training methods for the neural network
"""

import sys, time, os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from modules.dataset import *
from modules.utils import *
from modules.loss import TV_loss
from modules.network import Net


def imshow(img, target):

    npimg = make_grid(img).numpy()
    nptarget = make_grid(target).numpy()

    fig, ax = plt.subplots(2, 1, figsize=(10, 10),
                           sharex=True, sharey=True,
                           subplot_kw={'adjustable': 'box-forced'})

    ax[0].set_title("Original images")
    ax[0].imshow(np.transpose(npimg, (1, 2, 0)))

    ax[1].set_title("Target images")
    ax[1].imshow(np.transpose(nptarget, (1, 2, 0)))

    for a in ax.ravel():
        a.set_axis_off()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    # Select a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # Training parameters
    batch_size = 32
    num_epochs = 80
    learning_rate = 0.0002

    # Load the segmentation dataset
    segmentation_dataset = SegmentationDataset(
        root_dir='./data/',
        input_dir='train2017/',
        target_dir='trainSP2017/',
        transform=transforms.Compose([
             RandomCrop(224),
             Normalize(),
             ToTensor()]))

    # Data loader
    trainloader = torch.utils.data.DataLoader(
        segmentation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)

    # Neural network
    d = 7
    net = Net(d)
    net.to(device)
    run_idx = '14'
    PATH = './results/weights/run' + run_idx + '.pth'
    TEMP = './results/weights/run'

    net.load_state_dict(torch.load('./results/weights/run10_13.pth'))
    start_epoches = 0

    # Optimizer
    criterion = lambda outputs, target: TV_loss(outputs, target,
                                                batch_size=batch_size,
                                                alpha=0.)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    def learning():

        loss_lst = [[] for i in range(start_epoches, num_epochs)]

        # loop over the dataset multiple times
        for epoch in range(start_epoches, num_epochs):

            x = []
            running_loss = 0.0

            param_a = [[] for s in range(2, d-2+1)]
            param_b = [[] for s in range(2, d-2+1)]

            for i, sample in enumerate(trainloader):

                # Load batch
                input_img = sample['input'].to(device)
                segm_true = sample['target'].to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward/Backward pass
                outputs = net(input_img)
                loss = criterion(outputs, segm_true)
                loss.backward()

                # Weights update
                optimizer.step()
                running_loss += loss.item()
                if i % 1 == 0:

                    print('[%d,%d] loss: %.6f' % (epoch+1, i+1, running_loss/(i+1)))
                    loss_lst[epoch - start_epoches].append(loss.item())
                    x.append(i+1)

                    for s in range(2, d-2+1):
                        param_a[s-2].append(net.convs[s-2].ABN.a.data.item())
                        param_b[s-2].append(net.convs[s-2].ABN.b.data.item())

            # Save weights
            torch.save(net.state_dict(), TEMP + run_idx + "_" + str(epoch) + '.pth')

            # Update learning rate
            # if (epoch == 10):
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = 0.001

            # Save loss for the epoch
            np.save('./results/loss-train/run' + run_idx + "_" + str(epoch) + '.npy',
                    np.array(loss_lst[epoch - start_epoches]))

    learning()
    torch.save(net.state_dict(), PATH)
