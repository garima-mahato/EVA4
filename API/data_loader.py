from __future__ import print_function
import math
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from torchsummary import summary
from tqdm import tqdm
from albumentations import *
from albumentations.pytorch import ToTensor

# custom transformations from albumentations library
class AlbumentationTransformations():
  def __init__(self, means, stdevs):
    self.means = numpy.array(means)
    self.stdevs = numpy.array(stdevs)
    patch_size = 28
    self.album_transforms = Compose([
      RandomSizedCrop((patch_size,patch_size), patch_size,patch_size),
      HorizontalFlip(p = 0.5),
	  Cutout(num_holes=1, max_h_size=16, max_w_size=16, p=0.75),
      Normalize(mean=means, std=stdevs),
      ToTensor()
    ])
        
  def __call__(self, img):
      img = numpy.array(img)
      img = self.album_transforms(image=img)['image']
      return img


def augment_data(means, stdevs):
  transformations = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(means, stdevs)
  ])
  return transformations

def generate_train_test_loader(data_path, SEED, means, stdevs, is_albumentation=False):
  # Train Phase transformations
  if is_albumentation:
    train_transforms = AlbumentationTransformations(means, stdevs)
  else:
    train_transforms = augment_data(means, stdevs)

  # Test Phase transformations
  test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means, stdevs)
  ])

  train = datasets.CIFAR10(data_path, train=True, download=True, transform=train_transforms)
  test = datasets.CIFAR10(data_path, train=False, download=True, transform=test_transforms)

  # CUDA?
  cuda = torch.cuda.is_available()

  # For reproducibility
  torch.manual_seed(SEED)

  if cuda:
      torch.cuda.manual_seed(SEED)

  # dataloader arguments - something you'll fetch these from cmdprmt
  dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

  # train dataloader
  train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

  # test dataloader
  test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

  return train_loader, test_loader, test
