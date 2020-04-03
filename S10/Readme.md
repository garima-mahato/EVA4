# Classification on CIFAR10 using ResNet18 with hyperparameter tuning
---

[Github Link to code](https://github.com/genigarus/EVA4/blob/master/S10/EVA4_Session10.ipynb)

[Colab Link to code](https://colab.research.google.com/drive/1-ciM_PB0V9dACZEOxtz7BLjbzD18bpIM)

### 1) Image Augmentation used with CutOut being one of the transformation

![Image Augmentation](https://raw.githubusercontent.com/genigarus/EVA4/master/S10/visualization/transformations.PNG)

### 2) Found the best Learning Rate to train the model

![Best LR](https://raw.githubusercontent.com/genigarus/EVA4/master/S10/visualization/lr_finder.PNG)

**Best Learning Rate: 0.01584893192461112**

### 3) Used **SGD with momentum** with momentum = 0.9

### 4) Trained the ResNet-18 model for 50 epochs

EPOCH: 1
Loss=2.15325665473938 Batch_id=390 Accuracy=43.85: 100%|██████████| 391/391 [02:34<00:00,  2.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0097, Accuracy: 5419/10000 (54.19%)

EPOCH: 2
Loss=2.012441873550415 Batch_id=390 Accuracy=60.02: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0077, Accuracy: 6524/10000 (65.24%)

EPOCH: 3
Loss=1.6384177207946777 Batch_id=390 Accuracy=67.32: 100%|██████████| 391/391 [02:34<00:00,  2.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0082, Accuracy: 6519/10000 (65.19%)

EPOCH: 4
Loss=1.7387962341308594 Batch_id=390 Accuracy=71.24: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0064, Accuracy: 7264/10000 (72.64%)

EPOCH: 5
Loss=1.4806256294250488 Batch_id=390 Accuracy=73.87: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0069, Accuracy: 7144/10000 (71.44%)

EPOCH: 6
Loss=1.2457722425460815 Batch_id=390 Accuracy=75.62: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0052, Accuracy: 7764/10000 (77.64%)

EPOCH: 7
Loss=1.3323636054992676 Batch_id=390 Accuracy=77.13: 100%|██████████| 391/391 [02:34<00:00,  2.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0047, Accuracy: 7911/10000 (79.11%)

EPOCH: 8
Loss=1.1756545305252075 Batch_id=390 Accuracy=78.16: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0043, Accuracy: 8188/10000 (81.88%)

EPOCH: 9
Loss=1.0852481126785278 Batch_id=390 Accuracy=79.02: 100%|██████████| 391/391 [02:35<00:00,  2.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0041, Accuracy: 8255/10000 (82.55%)

EPOCH: 10
Loss=1.4269263744354248 Batch_id=390 Accuracy=79.67: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0037, Accuracy: 8453/10000 (84.53%)

EPOCH: 11
Loss=0.9477803707122803 Batch_id=390 Accuracy=80.62: 100%|██████████| 391/391 [02:34<00:00,  2.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0043, Accuracy: 8126/10000 (81.26%)

EPOCH: 12
Loss=0.9541343450546265 Batch_id=390 Accuracy=81.57: 100%|██████████| 391/391 [02:33<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0042, Accuracy: 8166/10000 (81.66%)

EPOCH: 13
Loss=0.9508116841316223 Batch_id=390 Accuracy=81.75: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0042, Accuracy: 8194/10000 (81.94%)

EPOCH: 14
Loss=0.907364010810852 Batch_id=390 Accuracy=82.21: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0048, Accuracy: 8018/10000 (80.18%)

EPOCH: 15
Loss=0.833751916885376 Batch_id=390 Accuracy=82.52: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0037, Accuracy: 8390/10000 (83.90%)

EPOCH: 16
Loss=0.6455802917480469 Batch_id=390 Accuracy=82.90: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0036, Accuracy: 8447/10000 (84.47%)

EPOCH: 17
Loss=0.9995672702789307 Batch_id=390 Accuracy=83.17: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0034, Accuracy: 8517/10000 (85.17%)

EPOCH: 18
Loss=0.872093141078949 Batch_id=390 Accuracy=83.89: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0035, Accuracy: 8450/10000 (84.50%)

EPOCH: 19
Loss=0.8379157781600952 Batch_id=390 Accuracy=83.82: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0036, Accuracy: 8513/10000 (85.13%)

EPOCH: 20
Loss=0.9011598825454712 Batch_id=390 Accuracy=84.30: 100%|██████████| 391/391 [02:35<00:00,  2.51it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0035, Accuracy: 8483/10000 (84.83%)

EPOCH: 21
Loss=0.7680745124816895 Batch_id=390 Accuracy=84.55: 100%|██████████| 391/391 [02:34<00:00,  2.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0043, Accuracy: 8184/10000 (81.84%)

EPOCH: 22
Loss=0.6993612051010132 Batch_id=390 Accuracy=84.64: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0033, Accuracy: 8578/10000 (85.78%)

EPOCH: 23
Loss=0.7411786317825317 Batch_id=390 Accuracy=84.69: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0034, Accuracy: 8553/10000 (85.53%)

EPOCH: 24
Loss=0.7328760027885437 Batch_id=390 Accuracy=85.22: 100%|██████████| 391/391 [02:34<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0034, Accuracy: 8571/10000 (85.71%)

EPOCH: 25
Loss=0.7580845355987549 Batch_id=390 Accuracy=85.41: 100%|██████████| 391/391 [02:34<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0030, Accuracy: 8702/10000 (87.02%)

EPOCH: 26
Loss=0.7757035493850708 Batch_id=390 Accuracy=85.39: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0030, Accuracy: 8689/10000 (86.89%)

EPOCH: 27
Loss=0.5850611925125122 Batch_id=390 Accuracy=85.91: 100%|██████████| 391/391 [02:35<00:00,  2.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0038, Accuracy: 8449/10000 (84.49%)

EPOCH: 28
Loss=0.6602646112442017 Batch_id=390 Accuracy=85.84: 100%|██████████| 391/391 [02:35<00:00,  2.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0030, Accuracy: 8726/10000 (87.26%)

EPOCH: 29
Loss=0.708653450012207 Batch_id=390 Accuracy=86.08: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0035, Accuracy: 8514/10000 (85.14%)

EPOCH: 30
Loss=0.8008662462234497 Batch_id=390 Accuracy=86.35: 100%|██████████| 391/391 [02:34<00:00,  2.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0028, Accuracy: 8833/10000 (88.33%)

EPOCH: 31
Loss=0.636769711971283 Batch_id=390 Accuracy=86.49: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0028, Accuracy: 8800/10000 (88.00%)

EPOCH: 32
Loss=0.7092130184173584 Batch_id=390 Accuracy=86.39: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0029, Accuracy: 8792/10000 (87.92%)

EPOCH: 33
Loss=0.752608060836792 Batch_id=390 Accuracy=86.69: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0036, Accuracy: 8503/10000 (85.03%)

EPOCH: 34
Loss=1.0032578706741333 Batch_id=390 Accuracy=86.82: 100%|██████████| 391/391 [02:34<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0031, Accuracy: 8683/10000 (86.83%)

EPOCH: 35
Loss=0.7567118406295776 Batch_id=390 Accuracy=86.98: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0025, Accuracy: 8964/10000 (89.64%)

EPOCH: 36
Loss=0.7628611326217651 Batch_id=390 Accuracy=87.04: 100%|██████████| 391/391 [02:34<00:00,  2.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0025, Accuracy: 8957/10000 (89.57%)

EPOCH: 37
Loss=0.7143936157226562 Batch_id=390 Accuracy=87.22: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0034, Accuracy: 8568/10000 (85.68%)

EPOCH: 38
Loss=0.6786922812461853 Batch_id=390 Accuracy=87.30: 100%|██████████| 391/391 [02:34<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0027, Accuracy: 8850/10000 (88.50%)

EPOCH: 39
Loss=0.797775387763977 Batch_id=390 Accuracy=87.53: 100%|██████████| 391/391 [02:33<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0027, Accuracy: 8826/10000 (88.26%)

EPOCH: 40
Loss=0.7024500966072083 Batch_id=390 Accuracy=87.49: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0036, Accuracy: 8526/10000 (85.26%)

EPOCH: 41
Loss=0.6100055575370789 Batch_id=390 Accuracy=87.59: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0025, Accuracy: 8939/10000 (89.39%)

EPOCH: 42
Loss=0.5923382043838501 Batch_id=390 Accuracy=87.90: 100%|██████████| 391/391 [02:35<00:00,  2.51it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0031, Accuracy: 8674/10000 (86.74%)

EPOCH: 43
Loss=0.6672704815864563 Batch_id=390 Accuracy=87.78: 100%|██████████| 391/391 [02:34<00:00,  2.54it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0028, Accuracy: 8787/10000 (87.87%)

EPOCH: 44
Loss=0.5858694911003113 Batch_id=390 Accuracy=87.72: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0028, Accuracy: 8819/10000 (88.19%)

EPOCH: 45
Loss=0.5993590950965881 Batch_id=390 Accuracy=88.05: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0027, Accuracy: 8878/10000 (88.78%)

EPOCH: 46
Loss=0.480996310710907 Batch_id=390 Accuracy=87.98: 100%|██████████| 391/391 [02:34<00:00,  2.52it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0026, Accuracy: 8868/10000 (88.68%)

EPOCH: 47
Loss=0.4863784909248352 Batch_id=390 Accuracy=91.49: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0017, Accuracy: 9288/10000 (92.88%)

EPOCH: 48
Loss=0.5819729566574097 Batch_id=390 Accuracy=92.66: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0016, Accuracy: 9330/10000 (93.30%)

EPOCH: 49
Loss=0.4391074776649475 Batch_id=390 Accuracy=93.26: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
  0%|          | 0/391 [00:00<?, ?it/s]
Test set: Average loss: 0.0016, Accuracy: 9347/10000 (93.47%)

EPOCH: 50
Loss=0.48527491092681885 Batch_id=390 Accuracy=93.74: 100%|██████████| 391/391 [02:34<00:00,  2.53it/s]
Test set: Average loss: 0.0015, Accuracy: 9367/10000 (93.67%)

### 5) Training Vs Test Accuracy

![Train vs Test Acc](https://raw.githubusercontent.com/genigarus/EVA4/master/S10/visualization/train_vs_test_acc_graph.png)

### 6) Graphs of Loss and Accuracy for training and testing

![Loss and Acc](https://raw.githubusercontent.com/genigarus/EVA4/master/S10/visualization/train_test_acc_loss_separate_graph.PNG)

### 7) Test Accuracy reached: 93.67%

### 8) GradCAM on 25 misclassified images

![25 misclassified images with gradcam](https://raw.githubusercontent.com/genigarus/EVA4/master/S10/visualization/gradcam_misclassified_images.png)


