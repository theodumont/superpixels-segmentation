"""
Neural network implementation
"""

import torch
import torch.nn as nn


# 1. ADAPTATIVE BATCH NORMALIZATION

class AdaptiveBatchNorm2d(nn.Module):

    """
    Adaptative batch normalization implementation
    """
    def __init__(self, num_features, momentum=.1, eps=1e-5, affine=True):

        r"""
        Class constructor

        An adaptative batch normalization layer takes as input a tensor x and
        outputs a tensor y defined by

        .. math::
        y = a BN(x) + bx

        where BN is a batch normalization layer, and a and b are learnable parameters.

        .. math::
        BN(x) = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

        The shape of the input tensor is BxCxWxH where B is the number of imaes in each batch,
        C the number of features maps, W the width of the image and H the height of the image,
        respectively.

        :param num_features: Number of features map
        :param momentum: Parameter used by the batch normalizatio layer to compute the statistics
        :param eps: Value added to the denominator for ensuring stability
        :param affine: When set to True, indicates that the batch normalization
         layer has learnable affine parameters.

        :type num_features: int
        :type momentum: float
        :type eps: float
        :type affine: Boolean

        ..seealso:: Pytorch documentation for nn.BatchNorm2D

        """
        super(AdaptiveBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, momentum, eps, affine)
        tens_a = torch.FloatTensor(1, 1, 1, 1)
        tens_b = torch.FloatTensor(1, 1, 1, 1)
        tens_a[0, 0, 0, 0] = 1
        tens_b[0, 0, 0, 0] = 0
        self.a = nn.Parameter(tens_a)
        self.b = nn.Parameter(tens_b)

    def forward(self, x):

        """
        Forward pass in the adaptative batch normalization layer

        .. math::
        y = a BN(x) + bx

        where BN is a batch normalization layer, and a and b are learnable parameters.

        :param x: Input tensor, with size BxCxWxH
        :type x: PyTorch tensor

        :return: Transformed tensor
        :rtype: PyTorch tensor
        """

        return self.a * x + self.b * self.bn(x)


# 2. CONVOLUTION MODULE

class ChenConv(nn.Module):

    """
    Convolution module implementation: 2D dilated Convolution followed
    by an adaptative batch normalization layer and a leaky ReLU
    activation function.
    """

    def __init__(self, in_channels, out_channels, s):

        """
        Class constructor

        :param in_channels: Input tensor with size BxCxWxH
        :param out_channes: Output tensor with size BxCxWxH
        :param s: Dilation scale
        :type in_channels: PyTorch tensor
        :type out_channels: PyTorch tensor
        :type s: int
        """
        super(ChenConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3,
                              padding=2**(s-1), dilation=2**(s-1))
        self.ABN = AdaptiveBatchNorm2d(out_channels)
        self.LReLU = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        """
        Forward pass in the convolution module

        :param x: Input tensor, with size BxCxWxH
        :type x: PyTorch tensor

        :return: Transformed tensor
        :rtype: PyTorch tensor
        """
        return self.LReLU(self.ABN(self.conv(x)))


# 3. NEURAL NETWORK ARCHITECTURE

class Net(nn.Module):

    """
    Implementation of the neural network architecture
    """

    def __init__(self, d):

        """
        Neural network architecture
        """

        super(Net, self).__init__()

        # First convolution module
        self.first = ChenConv(3, 24, 1)

        # Intermediate convolution modules
        self.convs = nn.ModuleList([ChenConv(24, 24, s) for s in range(2, d-2+1)])
        self.penulti = ChenConv(24, 24, 1)

        # Final convolution
        self.conv_ulti = nn.Conv2d(24, 3, 1, padding=0, dilation=1)
        self.ABN_ulti = AdaptiveBatchNorm2d(3)

    def forward(self, x):

        """
        Forward pass in the convolutional network

        :param x: Input tensor with size Bx3xWxH
         (B: batch size, 3: number of channels, W: image width, H: image height)
        :type x: PyTorch tensor

        :return: Transformed tensor
        :rtype: PyTorch Tensor
        """

        x = self.first(x)
        for l in self.convs:
            x = l(x)
        x = self.penulti(x)
        x = (self.ABN_ulti(self.conv_ulti(x)))
        return x
