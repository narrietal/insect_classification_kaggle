# Project summary


The project summary gives on overall view of the methods and the development process for the finalised model. Additionally, the summary entails the intermediate steps of the progress toward final accuracy. 

## Data analysis
The dataset was explored to get a better overall undertanding of the data that was used for analysis.

Hence, multiple problems from the dataset were outlined accordingly, that could pose a negative effect for the future model performance:

1. Some images that don't contain insects. (section 3.2)
2. Some images between one class represent different insects. (section 3.2)
3. Some images are repeated. (section 3.2)

Subsequently, the distribution of images per class which showed a highly unbalanced dataset (section 3.3.1) and in the light of these results, different techniques were applied to solve the issue:

*   Implementation of splitting the dataset (0.9 training, 0.1 validation). After testing the model, there weren't significant changes in final the performance.

*   The application of different balancing techniques to solve the matter:

  *   1) Attempt was to oversample the classes with less number of images and 
at the same time undersampling the few classes with a high number of images. Hence, in order to avoid overfitting different data augmentation techniques were applied to the oversampling images. Finally, this method was deemed implausible since it overfits the model.

  *   2) Attempt of balancing the dataset was through using [WeightedRandomSampler](https://pytorch.org/docs/stable/data.htmin) the dataloader. In order to accomplish it, a weight was assigned for each class. The dataloader would use samples of the classes with higher weights (the classes with less images). After testing this method, it was discarded since it gave worse results in terms of validation accuracy.

## Why was the balancing technique unsuccesful with the model?

The test set was also unbalanced. It followed the same distribution as the training and validation sets, meaning there is a large dispersion between the number of images of different classes. (Section 6.2)

Therefore, the overall accuracy is greater when using an unbalanced dataset since the most tested classes are the ones that the network had trained the most.

Subsequently, the unbalanced models that were tested presented a higher overall accuracy than the balanced models suggesting that the test set used in Kaggle follows the same distribution pattern as the dataset used during training.

## Data pre-processing

Data pre-processing was applied to some data refining over the dataset images as it can be seen in the data transformers (Section 4.2). The changes over the images are the following:

*   Resize of images to 224x224 to suit the Resnet/Resnext model input
*   Application of data augmentation techniques on the images to increase the variety of images. Many different  combinations of augmentations were tried and only "RandomHorizontalFlip" was kept, since it yielded the best results.
* Image to tensor
* Normalise the images such that all the features would be on the same scale which speeds up the training phase.


## Model choice and hyperparameters
To reach a higher accuracy, the ensemble learning technique was applied with two models. The two models used for this technique were Resnet152 and Resnext101 architectures. 

Resnet152 and Resnext101 architectures offer great performance on multiclass image classification with deep convulational layers, which would allow the training of the model in shorter time durations (approximately 1 hour). Both models have low top-5 errors in the learning rate, which make both architectures efficient.    

It's worth mentioning, by using more models with the ensemble learning technique the overall accuracy could be even higher. Also, two models were chosen in order to save GPU time. All in all, the sum of the two probability vectors have the higher value of the overall combined vector. Thus, the model improved by 2,7% with the ensemble learning technique by combining Resnet152 and Resnext101. 

Below are the model hyperparameters that were refined after various iterations:


Hyperparameters for Resnet152:
- Criterion: Cross Entropy Loss 

- Optimizer: Stochastic Gradient Descent (Learning rate: 0.001 & Momentum: 0.9)

- Epochs: 5

- Batch size: 32
 
for Resnext101:
- Criterion: Cross Entropy Loss 

- Optimizer: Stochastic Gradient Descent (Learning rate: 0.001 & Momentum: 0.9)

- Epochs: 10

- Scheduler: step 4 / gamma 0.1

- Batch size: 32
