# CNN on MNIST Data

Link to colab : https://colab.research.google.com/drive/1bx-ets2msLdt_vfw2oI842EaHDs3AG5n

### Best Test Accuracy: 99.50%

### Model Test Accuracy: 99.48%

## Model architecture

The model consist of 2 convolution block and 1 output block as follows:

1) Block 1: It is a convolution block consisting of 2 convolution layer each of 8 filters with 3x3 convolution. Each convolution layer is followed by batch normalization and dropout of 0.1 . After the convolution layer is Max Pooling layer.

2) Block 2: It is a convolution block consisting of 2 convolution layer each of 16 filters with 3x3 convolution. Each convolution layer is followed by batch normalization and dropout of 0.1 . After the convolution layer is Max Pooling layer.

3) Output Block: It consist of 2 convolution layer each of 16 and 32 filters respectively. Then a 1x1 convolution to reduce the number of channels to 10 followed by another 3x3 convolution of 10 filters.

**Steps taken to arrive at the final model**

1) Using the base model, I removed relu from the output layer because it was preventing negative values from reaching the prediction layer. This improved the accuracy to around 96%.

2) Then I reduced the number of filters and tried with different numbers until I brought down the number of parameters below 20k. While doing this, I tried to maintain the framework of 2 blocks of 2 convolution layer each. I got accuracy of around 98.6% and was able to bring the number of parameters to around 12k.

3) To further increase the accuracy, I used data augmentation which enhanced the validation accuracy to around 99.34% .

4) I noticed that there was quite some difference between train and test loss. So, I used dropout with 0.1 and batch normalization. Finally, I got accuracy of 99.48%.

## Model Summary

![https://raw.githubusercontent.com/genigarus/Assets/master/Images/EVA4/S4_model_summary.PNG](https://raw.githubusercontent.com/genigarus/Assets/master/Images/EVA4/S4_model_summary.PNG)


## Logs


Epoch 1/20


Epoch=1 loss=0.23246441781520844 batch_id=468: 100%|██████████| 469/469 [00:27<00:00, 17.02it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
  
Test set: Average loss: 0.0646, Accuracy: 9789/10000 (97.89%)



Epoch 2/20


Epoch=2 loss=0.08990276604890823 batch_id=468: 100%|██████████| 469/469 [00:26<00:00, 17.47it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
  
Test set: Average loss: 0.0331, Accuracy: 9900/10000 (99.00%)



Epoch 3/20


Epoch=3 loss=0.13187465071678162 batch_id=468: 100%|██████████| 469/469 [00:26<00:00, 17.38it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
  
Test set: Average loss: 0.0301, Accuracy: 9901/10000 (99.01%)



Epoch 4/20


Epoch=4 loss=0.10323920845985413 batch_id=468: 100%|██████████| 469/469 [00:26<00:00, 17.07it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
  
Test set: Average loss: 0.0278, Accuracy: 9904/10000 (99.04%)



Epoch 5/20


Epoch=5 loss=0.07981861382722855 batch_id=468: 100%|██████████| 469/469 [00:26<00:00, 17.75it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
  
Test set: Average loss: 0.0265, Accuracy: 9914/10000 (99.14%)



Epoch 6/20


Epoch=6 loss=0.2029215544462204 batch_id=468: 100%|██████████| 469/469 [00:26<00:00, 17.38it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
  
Test set: Average loss: 0.0242, Accuracy: 9924/10000 (99.24%)



Epoch 7/20


Epoch=7 loss=0.07452548295259476 batch_id=468: 100%|██████████| 469/469 [00:26<00:00, 17.62it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
  
Test set: Average loss: 0.0208, Accuracy: 9928/10000 (99.28%)



Epoch 8/20


Epoch=8 loss=0.04572306200861931 batch_id=468: 100%|██████████| 469/469 [00:27<00:00, 17.31it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
  
Test set: Average loss: 0.0244, Accuracy: 9929/10000 (99.29%)



Epoch 9/20


Epoch=9 loss=0.03571651875972748 batch_id=468: 100%|██████████| 469/469 [00:26<00:00, 17.65it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
  
Test set: Average loss: 0.0225, Accuracy: 9933/10000 (99.33%)



Epoch 10/20


Epoch=10 loss=0.05741700530052185 batch_id=468: 100%|██████████| 469/469 [00:27<00:00, 16.98it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
  
Test set: Average loss: 0.0234, Accuracy: 9924/10000 (99.24%)



Epoch 11/20


Epoch=11 loss=0.028517575934529305 batch_id=468: 100%|██████████| 469/469 [00:26<00:00, 17.42it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
  
Test set: Average loss: 0.0243, Accuracy: 9923/10000 (99.23%)



Epoch 12/20


Epoch=12 loss=0.07614418119192123 batch_id=468: 100%|██████████| 469/469 [00:28<00:00, 16.66it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
  
Test set: Average loss: 0.0209, Accuracy: 9925/10000 (99.25%)



Epoch 13/20


Epoch=13 loss=0.054434746503829956 batch_id=468: 100%|██████████| 469/469 [00:26<00:00, 17.48it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
  
Test set: Average loss: 0.0192, Accuracy: 9935/10000 (99.35%)



Epoch 14/20


Epoch=14 loss=0.05982019379734993 batch_id=468: 100%|██████████| 469/469 [00:27<00:00, 16.75it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
  
Test set: Average loss: 0.0213, Accuracy: 9934/10000 (99.34%)



Epoch 15/20


Epoch=15 loss=0.030997460708022118 batch_id=468: 100%|██████████| 469/469 [00:26<00:00, 17.51it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
  
Test set: Average loss: 0.0195, Accuracy: 9932/10000 (99.32%)



Epoch 16/20


Epoch=16 loss=0.11825542896986008 batch_id=468: 100%|██████████| 469/469 [00:27<00:00, 14.40it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
  
Test set: Average loss: 0.0183, Accuracy: 9945/10000 (99.45%)


Epoch 17/20

Epoch=17 loss=0.0329522080719471 batch_id=468: 100%|██████████| 469/469 [00:27<00:00, 17.34it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0163, Accuracy: 9950/10000 (99.50%)



Epoch 18/20


Epoch=18 loss=0.03957851231098175 batch_id=468: 100%|██████████| 469/469 [00:27<00:00, 17.04it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
  
Test set: Average loss: 0.0187, Accuracy: 9939/10000 (99.39%)



Epoch 19/20


Epoch=19 loss=0.016662806272506714 batch_id=468: 100%|██████████| 469/469 [00:27<00:00, 17.16it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
  
Test set: Average loss: 0.0190, Accuracy: 9943/10000 (99.43%)



Epoch 20/20


Epoch=20 loss=0.03207915648818016 batch_id=468: 100%|██████████| 469/469 [00:27<00:00, 17.14it/s]

Test set: Average loss: 0.0163, Accuracy: 9948/10000 (99.48%)
