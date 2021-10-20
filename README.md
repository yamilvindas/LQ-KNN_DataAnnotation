# Semi-automatic data annotation based on feature space projection and local quality metrics: An application to Cerebral Emboli characterization

## I) Introduction

This repository presents the code of the MNIST experiments of the submitted paper *Semi-automatic data annotation based on feature space projection and local quality metrics: An application to Cerebral Emboli characterization*.


## II) Proposed Method

![plot](./figs/DataAnnotationMethod.png)

Our proposed method is composed of four steps:
* **Feature Extraction**:  we start by extracting features in an unsupervised manner using an Autoencoder adapted to our data. Using unsupervised learning techniques allows to avoid handcrafted features and to use all the available samples (labeled and unlabeled). 
* **Dimensionality Reduction**: we reduce the dimension of the latent space of the previous step to obtain a 2D space. This allows to do a more efficient automatic and manual labeling of the samples as showed in [Benato et al. (2020)](
https://doi.org/10.1016/j.patcog.2020.107612). In this step, we compute different projections and we select the optimal projection using the *Silhouette Score*.
* **Manual annotation**: From the selected projectin in the previous step, a manual annotation is done in order to cover the whole annotation space. This step **is not tested** in this Git repository.
* **Automatic label propagation**: by considering the local projection quality of each sample in the 2D space ([Lueks et al. (2011)](https://arxiv.org/abs/1110.3917)), we propagate the labels of high quality labeled samples to high quality unlabeled samples. This allows to create a richer training-set with reduced effort.

Finally, once we obtained the final semi-automatically labeled dataset, we propose to do supervised classification using a robust loss function to compensante the noise introduced by the automatic label propagation.


![plot](./figs/DimensionalityReduction.png)


![plot](./figs/LQ-KNN_Principle.gif)

## III) Code Structure

The code is structured in different folders:
* **datasets**: This folder will store the different datasets used to test our proposed semi-automatic data annotation method.
* **src**: This folder contains the source codes necessary to run the different examples. In this folder we can find the code *label_propagation.py* which implements our proposed method. More details about this folder can be found in the README.md file of this folder.
* **examples**: This folder contains different examples corresponding to different experiments that we perform in our submitted paper. For more details refer to the README.md file in the folder.
* **parameters_files**: This folder contains json files with the parameters of different experiments. By default there is a file with the train parameters of the auto-encoder for feature extraction (first step of our method) and another file with the parameters to train a classification model using a semi-automatically labeled dataset.
* **models**: This folder will store the different models obtained with the source codes and their results. For example, in this folder you can find the trained auto-encoders and the results of label-propagation using that auto-encoder.
* **utils**: This folder contains some useful codes to plot the 2D projections, the training  metrics of the auto-encoders, the annotation performances of a label propagation method, etc.
* **figs**: This fodler contains the different figures used as illustrations in this Git repository.

## IV) Examples

### A) General Example

To execute a simple experiment doing label propagation you can simply execute the code "label_propagation.py" in the "examples" folder: 

**python ./examples/label_propagation.py**

This code will generate two figures for our proposed method with $\tau=0.1$:
*  One showing the annotation accuracy with respect to the considered neighborhood used for label propagation.
*  Another one showing the number of labeled samples with respect to the considered neighborhood used for label propagation.

![plot](./figs/LabelPropagationExample/annotation_accuracy.png)
![plot](./figs/LabelPropagationExample/nb_labeled_samples.png)

### B) Other Examples

In the folder *examples* you can find some codes testing the main blocks of our proposed method. For more details, you can refer to the README.md file of this folder.


