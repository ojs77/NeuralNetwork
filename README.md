# NeuralNetwork
I created a convolutional neural network to categories images from the CIFAR10 dataset. This dataset contains over 60,000 images of 10 different types of object.

There is a tutorials folder which shows the process I underwent to learn the techniques applied.

In the CNN final product folder there is all the files required to train and evaluate the model including the model itself. I built an original model taking inspiration from posts online and then improved upon this model using methods I have researched.
The accuracy of the improved model jumped from 46% on the original to 75%. More statistics can be found below.

| Original Model | Accuracy | PPV | F1-Score | \- | Improved Model | Accuracy | PPV | F1-Score |
| -------------- | -------- | --- | -------- | -- | -------------- | -------- | --- | -------- |
| Bird           | 27%      | 41% | 33%      | \- | Bird           | 55%      | 73% | 63%      |
| Car            | 51%      | 68% | 58%      | \- | Car            | 84%      | 91% | 87%      |
| Cat            | 15%      | 37% | 22%      | \- | Cat            | 52%      | 61% | 56%      |
| Deer           | 27%      | 47% | 34%      | \- | Deer           | 71%      | 70% | 70%      |
| Dog            | 55%      | 37% | 44%      | \- | Dog            | 68%      | 64% | 66%      |
| Frog           | 64%      | 50% | 56%      | \- | Frog           | 82%      | 80% | 81%      |
| Horse          | 67%      | 41% | 51%      | \- | Horse          | 86%      | 71% | 78%      |
| Plane          | 49%      | 50% | 50%      | \- | Plane          | 76%      | 80% | 78%      |
| Ship           | 67%      | 46% | 55%      | \- | Ship           | 88%      | 77% | 83%      |
| Truck          | 42%      | 56% | 48%      | \- | Truck          | 87%      | 80% | 83%      |
| Overall        | 46%      | 47% | 45%      | \- | Overall        | 75%      | 75% | 74%      |
