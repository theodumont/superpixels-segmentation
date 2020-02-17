"""
Load pairs of image and target from the dataset, and define
transformations on the images (crop, rescaling, etc.)
"""

import torch
import skimage.transform as tr
from skimage import img_as_float
import os
from skimage import io
import numpy as np


class SegmentationDataset:

    """
    Class handling the segmentation dataset
    """

    def __init__(self, root_dir, input_dir, target_dir, transform = None):

        """
        Class constructor

        :param root_dir: Root directory 
        :type root_dir: string
        :param input_dir: path to the training images
        :type input_dir: string
        :param target_dir: path to the training targets
        :type target_dir: string
        :param transform: Transformations to apply to the images
        :type transform: Python list
        """

        self.root_dir = root_dir
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.images = os.listdir(self.root_dir + self.target_dir)


    def __len__(self):

        """
        Return the number of images in the dataset

        :return: Number of images in the dataset
        :rtype: int
        """

        return len(self.images)


    def __getitem__(self, idx):

        """
        Return an image of the dataset along with the corresponding
        target image

        :param idx: Index of the image in the dataset
        :type idx: int

        :return: Dictionary containing the input image (key: "input")
         and the target image (key: "target")
        :rtype: Dictionary
        """

        img_name = os.path.join(self.root_dir,
          self.input_dir,
          self.images[idx][:-4] + '.jpg')

        target_name = os.path.join(self.root_dir,
          self.target_dir,
          self.images[idx])

        img = io.imread(img_name)
        target = io.imread(target_name)

        sample = {'input': img, 'target': target}

        if self.transform:
            sample = self.transform(sample)

        return sample



class Rescale(object):

    """
    Rescale images from a sample to a specified size.

    Args:

      output_size (tuple or int): Desired output size. 
      If tuple, the output shape matches output_size. 
      If int, the smallest edge size is matched to output_size,
      and the aspect ratio is kept identical

    """

    def __init__(self, output_size):

        """
        Constructor

        :param output_size: Desired output size. 
         If tuple, the output shape matches output_size. 
         If int, the smallest edge size is matched to output_size,
         and the aspect ratio is kept identical

        :type output_size: tuple or int
        """

        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        """
        Apply the transform to the specified sample

        :param sample: Dictionary containing the input image (key: "input")
         and the target image (key: "target")
        :type sample: Dictionary

        :return: Dictionary containing the transformed input image (key: "input")
         and the target image (key: "target")
        :rtype: Dictionary
        """

        img, target = sample['input'], sample['target']

        h, w = img.shape
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = tr.resize(img, (new_h, new_w))
        target = tr.resize(target, (new_h, new_w))

        return {'img': img, 'target': target}


class Crop(object):

    """
    Crop images from a sample

    Args:

      output_size (tuple or int): Desired output size. If int, a square crop
      is extracted.

    """

    def __init__(self, output_size):

        """
        Constructor

        :param output_size: Desired output size. If int, a square crop
         is extracted.
        :type output_size: int
        """

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)

        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        """
        Apply the transform to the specified sample

        :param sample: Dictionary containing the input image (key: "input")
         and the target image (key: "target")
        :type sample: Dictionary

        :return: Dictionary containing the transformed input image (key: "input")
         and the target image (key: "target")
        :rtype: Dictionary
        """

        img, target = sample['image'], sample['target']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = 0
        left = 0

        img = img[top: top + new_h,
                      left: left + new_w]

        target = target[top: top + new_h,
                      left: left + new_w]

        return {'input': image, 'target': image_segm}



class RandomCrop(object):

    """
    Randomly crop images from a sample.

    Args:

      output_size (tuple or int): Desired output size. If int, a square crop
      is extracted.

    """

    def __init__(self, output_size):

        """
        Constructor

        :param output_size: Desired output size. If int, a square crop
         is extracted.
        :type output_size: int
        """

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)

        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):

        """
        Apply the transform to the specified sample

        :param sample: Dictionary containing the input image (key: "input")
         and the target image (key: "target")
        :type sample: Dictionary

        :return: Dictionary containing the transformed input image (key: "input")
         and the target image (key: "target")
        :rtype: Dictionary
        """

        img, target = sample['input'], sample['target']

        h, w = img.shape[:2]
        new_h, new_w = self.output_size
        if(h - new_h < 0):
            print("Probleme h")
            print(h)
            print(new_h)
        if(w - new_w < 0):
            print("probleme w")
            print(w)
            print(new_w)

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = img[top: top + new_h,
                      left: left + new_w]
        target = target[top: top + new_h,
                      left: left + new_w]

        return {'input': img, 'target': target}


class ToTensor(object):

    """
    Converts the ndarrays from the samples into PyTorch Tensors
    """

    def __call__(self, sample):

        """
        Apply the transform to the specified sample

        :param sample: Dictionary containing the input image (key: "input")
         and the target image (key: "target")
        :type sample: Dictionary

        :return: Dictionary containing the transformed input image (key: "input")
         and the target image (key: "target")
        :rtype: Dictionary
        """

        img, target = sample['input'], sample['target']

        img = img.transpose((2, 0, 1))
        target = target.transpose((2, 0, 1))

        img = torch.from_numpy(img).float()
        target = torch.from_numpy(target).float()
        return {'input': img, 'target': target}


class Normalize(object):

    """
    Normalizes images from a sample between 0 and 1
    """

    def __call__(self, sample):

        """
        Apply the transform to the specified sample

        :param sample: Dictionary containing the input image (key: "input")
         and the target image (key: "target")
        :type sample: Dictionary

        :return: Dictionary containing the transformed input image (key: "input")
         and the target image (key: "target")
        :rtype: Dictionary
        """

        img, target = sample['input'], sample['target']
        img = img_as_float(img)
        target = img_as_float(target)
        return {'input': img, 'target': target}

