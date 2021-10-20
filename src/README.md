# Source codes

This folder contains the source codes necessary to run the different examples:
* **ae_model.py**: Defines the auto-encoder architecture used for feature extraction.
* **classification_model.py**: Defines the Convolutional Neural Network architecture used for classification.
* **feature_extraction.py**: Performs feature extraction using the model in *ae_model.py*. This means that it trains an auto-encoder model using the MNIST dataset.
* **label_propagation.py**: Implements our label propagation method and other propagation methods that can be used for comparison.
* **optimal_projection_selection.py**: Implements our proposed projection selection strategy based on the *Silhouette Score*.
* **projection_metrics.py**: Implements some projection metrics such as *local quality*, *quality from Lueks et al. (2011)*, *trustworthiness*, *continuity* and *LCMC*. **WARNING**: Some of the metrics can take some time to be computed.
* **tsne_grid_search.py**: Does a grid search over some pre-defined hyper-parameters to obtain different t-SNE projections. **WARNING**: the execution of this code can be take a considerable amount of time (a couple of hours). If you want to reduce the execution time, you can modify the grid search parameters inside the code.

More details about the different arguments of each code and the generated files can be found inside the codes.
