# dl_with_torch

### Contents

* main.py
  * This file has training and test functions 

* models folder
  * has all CNN models that are  ready to be trained
  * resnet.py
  * custom_resnet.py
  
* utils folder
  *  Files in this folder have required utility functions.
  * utils.py
  
* src folder
  * Files in this folder have required functions for calculating classwise accuracy, creatingcustom dataset classes ,performing  data augmentation ,
    identifying misclassifed images,plotting graphs for train and test accuracy and lossses.
  * accuracy.py
  * custom_dataset.Cifar10.py
  * custom_dataset_tiny_imagenet.py
  * data_augmentation.py
  * misclassification.py
  * plot_train_test_accuracy.py
  * subset.py
  
 * src/gradcam folder
   * Files in this folder have required functions for gradcam calculation and visualization
   * gradcam.py
   * visualization.py

