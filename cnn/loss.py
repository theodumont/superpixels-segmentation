"""
Implementation of the MSE + TV Loss function
"""

import torch
import torch.nn as nn


def TV_loss(outputs, target, batch_size, alpha=0.):

    """
    Implementation of the TV-loss function

    :param outputs: Output image of the neural network
    :param target: Target image
    :param batch_size: Batch size
    :param alpha: TV loss parameter (Default:0)

    :type outputs: PyTorch tensor
    :type target: PyTorch tensor
    :type batch_size: int
    :type alpha: float

    :return: Total variation loss
    :rtype: float
    """

    MSE = nn.MSELoss()
    loss = torch.sum(torch.abs(outputs[:, :, 1:, :] -
                               outputs[:, :, :-1, :]))
    loss += torch.sum(torch.abs(outputs[:, :, :, 1:] -
                                outputs[:, :, :, :-1]))
    return MSE(outputs, target) + alpha/batch_size/3 * loss
