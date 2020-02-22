# CNN on MNIST Data 
---

Link to colab : https://colab.research.google.com/drive/1ulqQlvEnpxvYdn47PX1DwzCwCf1HjGaQ


**Target**:

> Make test accuracy of 99.4% consistent by trying different learning rates and getting a proper LR. Based on logs of previous files, we see that the learning rate increases and decreases, i.e. it keeps oscillating, thus hinting that we are not using proper learning rate.

**Results**:

> Parameters: 10,008

> Best Train Accuracy: 98.57%

> Best Test Accuracy: 99.40%

**Analysis**:

> While smoothing the learning, I used StepLR and played with learning_rate, gamma and step_size.

> For learning_rate, 0.1 was good to start learning to reach the target accuracy. Lower starting values made learning slower while higher starting values made it to overshoot and disrupt the learning graph.

> For gamma, 0.5 was found to consistently descend towards minima.

> For step_size, 6 was sufficient to bring consistency. Lower values made learning to overshoot and hence never reaches 99.4% while higher values made the descending slower and not sufficient to reach within 15 epochs.

**Learning rate=0.1, gamma=0.5, step_size=6**
