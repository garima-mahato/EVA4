# 1) Triangular Cyclic Graph

[custom_cyclic_learning_rate](https://github.com/genigarus/EVA4/blob/master/S11/custom_cyclic_learning_rate.py) contains code for graph.

Below is the graph created by the code:
```
from custom_cyclic_learning_rate import *

lr_min, lr_max, step_size, max_iteration, path, name = 0.08, 0.8, 10, 100, PATH+"/visualization", "clr_graph"
generate_cyclic_learning_rate(lr_min, lr_max, step_size, max_iteration, path, name)
```

The entire code is in [EVA4_Session11_CLR](https://github.com/genigarus/EVA4/blob/master/S11/EVA4_Session11_CLR.ipynb) file.

![](https://raw.githubusercontent.com/genigarus/EVA4/master/S11/visualization/clr_graph.png)


# 2) Custom Residual Network

[Model Link](https://github.com/genigarus/EVA4/blob/master/API/models/CustomResidualNet.py)

LR range test was used to find maximum LR for one cycle policy by training model over 500 epochs

![](https://raw.githubusercontent.com/genigarus/EVA4/master/S11/visualization/lr_range_test_graph.PNG)


[Creating and training of model with hyperparameter tuning in 24 epochs with **90.08% test accuracy**](https://github.com/genigarus/EVA4/blob/master/S11/EVA4_Session11.ipynb)

**Training vs Test Accuracy**

![](https://raw.githubusercontent.com/genigarus/EVA4/master/S11/visualization/train_vs_test_acc_graph.png)


**GradCAM for 25 misclassified images**

![](https://raw.githubusercontent.com/genigarus/EVA4/master/S11/visualization/gradcam_misclassified_images.png)
