**Colab Link to notebook** - https://colab.research.google.com/drive/1WIf4kRDwKZSW-ir9_vLO9iTjXbTJK2l5

---

**Analysis of different decay rate values for L2 regularization**

![](https://raw.githubusercontent.com/genigarus/EVA4/master/S6/Assets/visualization/l2_val_loss_comparison.png)

![](https://raw.githubusercontent.com/genigarus/EVA4/master/S6/Assets/visualization/l2_val_acc_comparison.png)

From the above graphs, we can see that for decay rate of 0.00001 and 0.0001, the validation loss is quite low and very smooth. So, 0.00001(1e-5) is taken as decay rate value. 

---

**Effect of Regularization on Validation Accuracy**

![acc](https://raw.githubusercontent.com/genigarus/EVA4/master/S6/Assets/visualization/val_acc_comparison.png)

**Effect of Regularization on Validation Loss**

![loss](https://raw.githubusercontent.com/genigarus/EVA4/master/S6/Assets/visualization/val_loss_comparison.png)

From the validation loss comparison graph, we can see that adding any regularization L1(with lambda=1e-5) or L2(with decay rate=1e-5) smoothes out the loss curve by penalizing the weights. L2 starts showing lower loss value sooner in comparison to L1. Thus, for smaller number of epochs L2 would seem to work better. But, as we increase the number of epochs, both regularization show comparable results. For higher values of lambda in L1 regularization, validation loss is high as compared to without L1 regularization because the weights are penalized heavily taking into consideration the fact that the model is already quite small and simple. L1 and L2 regularization together makes the loss curve smoother and takes to a lower final loss value in comparison to all the four.

---

**25 Misclassified Images for L1 regularization**

![imgs](https://raw.githubusercontent.com/genigarus/EVA4/master/S6/Assets/visualization/l1_misclassified_images.png)

**25 Misclassified Images for L2 regularization**

![imgs](https://raw.githubusercontent.com/genigarus/EVA4/master/S6/Assets/visualization/l2_misclassified_images.png)
