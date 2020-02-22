# CNN on MNIST Data 
---

Link to colab : https://colab.research.google.com/drive/1J-987TBl3NdJoGsY90F6XpmfSQAztQ5O


**Target**:

> Increase starting train/test accuracy of model keeping parameters less than 10k. As Batch Normalization normalizes the channel/layer outputs, the internal channel outputs is no longer affected by contrasts and jitters in the images as we saw in previous file. It focuses only on edges and patterns. Thus, it would help in increasing the starting accuracy.

**Results**:

> Parameters: 10,056

> Best Train Accuracy: 99.53%

> Best Test Accuracy: 99.19%

**Analysis**:

> As we saw, batch normalization increased the beginning accuracy. So, now I was able to reach around 99% test accuracy. 

> We see that the number of parameters has exceeded 10k.

> There is also slight overfitting in training.
