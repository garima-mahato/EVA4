from __future__ import print_function
import math
import numpy
import torch
from torchvision import datasets, transforms
from .data_augmenter import *

def generate_train_test_loader(data_path, SEED, means, stdevs, is_albumentation=False, batch_size=128, is_augment=True):
  # Train Phase transformations
  if is_albumentation:
    train_transforms = AlbumentationTransformations(means, stdevs)
  elif is_augment:
    train_transforms = augment_data(means, stdevs)
  else:
    train_transforms = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize(means, stdevs)
	  ])

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
  dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

  # train dataloader
  train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

  # test dataloader
  test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

  return train_loader, test_loader
