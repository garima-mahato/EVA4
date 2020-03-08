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

def find_cifar10_normalization_values():
  num_of_inp_channels = 3
  simple_transforms = transforms.Compose([
                                        transforms.ToTensor()
                                       ])
  exp = datasets.CIFAR10('./data', train=True, download=True, transform=simple_transforms)
  data = exp.data
  data = data.astype(numpy.float32)/255
  means = ()
  stdevs = ()
  for i in range(num_of_inp_channels):
      pixels = data[:,:,:,i].ravel()
      means = means +(round(numpy.mean(pixels)),)
      stdevs = stdevs +(numpy.std(pixels),)

  print("means: {}".format(means))
  print("stdevs: {}".format(stdevs))
  print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))

  return means, stdevs

# visualize accuracy and loss graph
def visualize_graph(train_losses, train_acc, test_losses, test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc[4000:])
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")

def set_device():
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  return device

# view and save comparison graph of cal accuracy and loss
def visualize_save_comparison_graph(EPOCHS, dict_list, title, xlabel, ylabel, PATH, name="fig"):
  plt.figure(figsize=(20,10))
  epochs = range(1,EPOCHS+1)
  for label, item in dict_list.items():
    plt.plot(epochs, item, label=label)
  
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend()
  plt.savefig(PATH+"/visualization/"+name+".png")

# view and save misclassified images
def show_save_misclassified_images(model, test_loader, name="fig", max_misclassified_imgs=25):
  cols = 5
  rows = math.ceil(max_misclassified_imgs / cols)
  
  with torch.no_grad():
    ind = 0
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      misclassified_imgs_pred = pred[pred.eq(target.view_as(pred))==False]
      
      misclassified_imgs_data = data[pred.eq(target.view_as(pred))==False]
      misclassified_imgs_target = target[(pred.eq(target.view_as(pred))==False).view_as(target)]
      if ind == 0:
        example_data, example_targets, example_preds = misclassified_imgs_data, misclassified_imgs_target, misclassified_imgs_pred
      elif example_data.shape[0] < max_misclassified_imgs:
        example_data = torch.cat([example_data, misclassified_imgs_data], dim=0)
        example_targets = torch.cat([example_targets, misclassified_imgs_target], dim=0)
        example_preds = torch.cat([example_preds, misclassified_imgs_pred], dim=0)
      else:
        break
      ind += 1
    example_data, example_targets, example_preds = example_data[:max_misclassified_imgs], example_targets[:max_misclassified_imgs], example_preds[:max_misclassified_imgs]

  fig = plt.figure(figsize=(20,10))
  for i in range(max_misclassified_imgs):
    plt.subplot(rows,cols,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i].cpu().numpy(), cmap='gray', interpolation='none')
    plt.title(f"{i+1}) Ground Truth: {example_targets[i]},\n Prediction: {example_preds[i]}")
    plt.xticks([])
    plt.yticks([])
  plt.savefig(PATH+"/visualization/"+name+".png")
