# NeuralNetwork
I created a convolutional neural network to categories images from the CIFAR10 dataset. This dataset contains over 60,000 images of 10 different types of object.

There is a tutorials folder which shows the process I underwent to learn the techniques applied.

In the CNN final product folder there is all the files required to train and evaluate the model including the model itself. I built an original model taking inspiration from posts online and then improved upon this model using methods I have researched.

The accuracy of the improved model jumped from 46% on the original to 75%. More statistics can be found below. Accuracy indicates how often it correctly guesses the respective class when shown an image from that class. PPV (Positive predictive value) indicates if the model has predicted that class, how likely it's prediction is to be correct.


| Original Model | Accuracy | PPV | \- | Improved Model | Accuracy | PPV |
| -------------- | -------- | --- | -- | -------------- | -------- | --- |
| Bird           | 27%      | 41% | \- | Bird           | 55%      | 73% |
| Car            | 51%      | 68% | \- | Car            | 84%      | 91% |
| Cat            | 15%      | 37% | \- | Cat            | 52%      | 61% |
| Deer           | 27%      | 47% | \- | Deer           | 71%      | 70% |
| Dog            | 55%      | 37% | \- | Dog            | 68%      | 64% |
| Frog           | 64%      | 50% | \- | Frog           | 82%      | 80% |
| Horse          | 67%      | 41% | \- | Horse          | 86%      | 71% |
| Plane          | 49%      | 50% | \- | Plane          | 76%      | 80% |
| Ship           | 67%      | 46% | \- | Ship           | 88%      | 77% |
| Truck          | 42%      | 56% | \- | Truck          | 87%      | 80% |
| Overall        | 46%      | 47% | \- | Overall        | 75%      | 75% |
