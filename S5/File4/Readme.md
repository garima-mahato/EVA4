# CNN on MNIST Data 
---

Link to colab : https://colab.research.google.com/drive/1qPKo-Ys5iYg5Pon646tUGSg289I4ZbYn


**Target**:

> Add Image Augmentation to get a test accuracy target of 99.4% or above. I used RandomAffine which consiste of slight rotation, translation and scaling based on image analysis from the first file.

**Results**:

> Parameters: 10,008

> Best Train Accuracy: 98.57%

> Best Test Accuracy: 99.40%

**Analysis**:

> Despite adding dropout, we still see some overfitting because there are already very few parameters. So, I used a dropout of 0.

> I used random affine as data transformation which consist of slight rotation, translation and scaling. Image augmentation increased the test accuracy to 99.4%.  
