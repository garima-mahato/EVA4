# CNN on MNIST Data 
---

Link to colab : https://colab.research.google.com/drive/1THoV7fUXcQ1tdIHk5OMSCe2YOe-FhiqB


**Target**:

> Reduce number of parameters to about 10k and add dropout to handle overfitting.

**Results**:

> Parameters: 10,008

> Best Train Accuracy: 99.27%

> Best Test Accuracy: 99.38%

**Analysis**:

> I was able to bring down the number of parameters to about 10k by reducing the number of kernels and the gain in parameter space was filled by another 1x1 convolution layer(FC layer). Thus, balancing out the reduction in parameters without affecting test accuracy. 

> I added a small dropout of 0.05 to handle over-fitting because adding large values tended to decrease the test accuracy though decreasing difference between train and test accuracy. This value was kind of trade-off between decrease in difference between train and test accuracy, and keeping test accuracy around 99.3%.

> Reduction in number of kernels also reduces over-fitting.
