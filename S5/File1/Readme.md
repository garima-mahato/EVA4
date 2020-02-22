# CNN on MNIST Data 
---

Link to colab : https://colab.research.google.com/drive/1wFDEw5Y09pdLBZxq5hgP7W3FPTuUdlBz


**Target**:

> Analyse the data

> Create a basic skeleton of model using expand and squeeze model with GAP layer followed by FC layer(using 1x1) so that we get high accuracy in less than 10k parameters.

**Results**:

> Parameters: 9,876

> Best Train Accuracy: 98.56%

> Best Test Accuracy: 98.58%

**Analysis**:

> *Data Analysis*: 

>>mean and standard deviation are 0.1307 and 0.3081 respectively.

>> some images were slightly rotated, some shifted, some quite distinguishable while others were marked with light marks. These transformatins can be used for image augmentation.

> With 10k parameters, training up to 15 epochs led to test accuracy of around 98%. 

> We see that the starting accuracy is very low for such a small dataset like MNIST. This starting train/test accuracy has to be improved to reach higher accuracy within 15 epoch.
