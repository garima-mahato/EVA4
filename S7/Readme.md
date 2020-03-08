# CNN on CIFAR10
---

1) Change the code such that it uses GPU

2) Changed the architecture to C1C2C3C40 (basically 3 MPs)

3) Total RF (must be more than 44) = 102

4) One of the layers use Depthwise Separable Convolution

5) One of the layers use Dilated Convolution

6) Used GAP and added FC after GAP to target number of classes

7) Best Test Accuracy = 85.3%, Final Test accuracy = 84.73%, number of epochs = 50, Total Params = 906,528. 

8) Used image augmentation

[Link to Google Colab Code File](https://colab.research.google.com/drive/1gsmdBMLnRb7J9piNocCxsu-FcH8A1YoK)
