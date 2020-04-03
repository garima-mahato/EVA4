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

def find_cifar10_normalization_values(data_path='./data'):
  num_of_inp_channels = 3
  simple_transforms = transforms.Compose([
                                        transforms.ToTensor()
                                       ])
  exp = datasets.CIFAR10(data_path, train=True, download=True, transform=simple_transforms)
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

def visualize_save_train_vs_test_graph(EPOCHS, dict_list, title, xlabel, ylabel, PATH, name="fig"):
  plt.figure(figsize=(20,10))
  #epochs = range(1,EPOCHS+1)
  for label, item in dict_list.items():
    x = numpy.linspace(1, EPOCHS+1, len(item))
    plt.plot(x, item, label=label)
  
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.legend()
  plt.savefig(PATH+"/"+name+".png")

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
def classify_images(model, test_loader, device, max_imgs=25):
  misclassified_imgs = []
  correct_imgs = []
    
  with torch.no_grad():
    ind = 0
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

      misclassified_imgs_pred = pred[pred.eq(target.view_as(pred))==False]
      misclassified_imgs_indexes = (pred.eq(target.view_as(pred))==False).nonzero()[:,0]
      for mis_ind in misclassified_imgs_indexes:
        if len(misclassified_imgs) < max_imgs:
          misclassified_imgs.append({
              "target": target[mis_ind].cpu().numpy(),
              "pred": pred[mis_ind][0].cpu().numpy(),
              "img": data[mis_ind]
          })
    
	#for data, target in test_loader:
      correct_imgs_pred = pred[pred.eq(target.view_as(pred))==True]
      correct_imgs_indexes = (pred.eq(target.view_as(pred))==True).nonzero()[:,0]
      for ind in correct_imgs_indexes:
        if len(correct_imgs) < max_imgs:
          correct_imgs.append({
              "target": target[ind].cpu().numpy(),
              "pred": pred[ind][0].cpu().numpy(),
              "img": data[ind]
          })
      
  return misclassified_imgs, correct_imgs

def plot_images(images, PATH, name="fig", sub_folder_name="/visualization", is_cifar10 = True):
  cols = 5
  rows = math.ceil(len(images) / cols)
  fig = plt.figure(figsize=(20,10))

  for i in range(len(images)):
    img = denormalize(images[i]["img"])
    plt.subplot(rows,cols,i+1)
    plt.tight_layout()
    plt.imshow(numpy.transpose(img.cpu().numpy(), (1, 2, 0)), cmap='gray', interpolation='none')
    if is_cifar10:
      CIFAR10_CLASS_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
      plt.title(f"{i+1}) Ground Truth: {CIFAR10_CLASS_LABELS[images[i]['target']]},\n Prediction: {CIFAR10_CLASS_LABELS[images[i]['pred']]}")
    else:
      plt.title(f"{i+1}) Ground Truth: {images[i]['target']},\n Prediction: {images[i]['pred']}")
    plt.xticks([])
    plt.yticks([])
  plt.savefig(PATH+sub_folder_name+"/"+str(name)+".png")

def show_save_misclassified_images(model, test_loader, device, PATH, name="fig", max_misclassified_imgs=25):
  misclassified_imgs, _ = classify_images(model, test_loader, device, max_misclassified_imgs)
  plot_images(misclassified_imgs, PATH, name)

def show_save_correctly_classified_images(model, test_loader, device, PATH, name="fig", max_correctly_classified_images_imgs=25):
  _, correctly_classified_images = classify_images(model, test_loader, device, max_correctly_classified_images_imgs)
  plot_images(correctly_classified_images, PATH, name)

def denormalize(tensor, mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]):
  single_img = False
  if tensor.ndimension() == 3:
    single_img = True
    tensor = tensor[None,:,:,:]

  if not tensor.ndimension() == 4:
    raise TypeError('tensor should be 4D')

  mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
  std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
  ret = tensor.mul(std).add(mean)
  return ret[0] if single_img else ret

def imshow(img):
	img = denormalize(img)
	npimg = img.numpy()
	plt.imshow(numpy.transpose(npimg, (1, 2, 0)))