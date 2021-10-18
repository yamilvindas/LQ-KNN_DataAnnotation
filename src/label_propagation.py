#!/usr/bin/env python3
"""
    This code implements the different functions allowing to do label propagation
    from a small number of labeled samples to a large number of unlabeled samples
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neighbors import NearestNeighbors
import random
from time import time
import copy
from opfython.models import SemiSupervisedOPF

def getSamplesInformation(labeled_samples, unlabeled_samples):
    """
        Creates four lists with the samples, annotation status, labels and local qualities
        WARNING: We consider the unlabeled samples as having a label None

        Arguments:
        ----------
        labeled_samples: list
            List containing the labeled samples. Each sample (i.e. value of the
            list is a dictionary with THREE keys: 'Data', 'Label', 'LocalQuality')
        unlabeled_samples: dict
            Same as labeled_samples but in this case the 'Label' of each sample
            is None (except if the label is known and the label propagation is
            done for evaluation purposes)

        Returns:
        --------
        samples: list
            List of ALL the data of samples (labeled and unlabeled)
        annotation_status: list
            List of the annotation status of ALL the samples (labeled and
            unlabeled). If element i of the list is False, then it means that
            sample i is unlabeled.
        labels: list
            List of the labels of ALL the samples (labeled and unlabeled).
        local_qualities: list
            List of the local qualities of ALL the samples (labeled and unlabeled).
    """
    # Creating the lists
    samples, annotation_status, labels, local_qualities = [], [], [], []
    for sample in labeled_samples:
        samples.append(sample['Data'])
        annotation_status.append(True)
        labels.append(sample['Label'])
        local_qualities.append(sample['LocalQuality'])
    for sample in unlabeled_samples:
        samples.append(sample['Data'])
        annotation_status.append(False)
        labels.append(sample['Label'])
        local_qualities.append(sample['LocalQuality'])

    # Normalizing the local qualities
    maxLocalQuals = max(local_qualities)
    local_qualities = [localQual/maxLocalQuals for localQual in local_qualities]

    return samples,\
           annotation_status,\
           labels,\
           local_qualities


def propagateLabels_LQKNN(
                            labeled_samples,
                            unlabeled_samples,
                            K=10,
                            local_quality_threshold=0.1
                          ):
    """
        Does label propagation using LQ-KNN.

        Arguments:
        ----------
        labeled_samples: list
            List containing the labeled samples. Each sample (i.e. value of the
            list is a dictionary with THREE keys: 'Data', 'Label', 'LocalQuality')
        unlabeled_samples: dict
            Same as labeled_samples but in this case the 'Label' of each sample
            is None (except if the label is known and the label propagation is
            done for evaluation purposes)
        K: int
            Neighborhood to consider for label propagation
        local_quality_threshold: float
            Threshold used to define the 'good' quality samples. A sample is
            considered as 'good' if its local quality value is greater than
            the defined threshold

        Returns:
        --------
        new_annotated_samples: list
            List of the new annotated samples. Each sample is a dictionary with
            three keys: 'Data', 'Label', 'LocalQuality'
        accuracy_annotation: float or None
            If the true labels of the unlabeled samples are available, the
            annotation accuracy is computed. If not, its value is None
        nb_annotated_samples: int
            Total numbr of labeled samples
        total_number_of_samples: int
            Total number of samples. It may not be equal to
            nb_annotated_samples + labeled_samples because not all the
            unlabeled_samples are always labeled.
        number_initial_labeled_samples int
            Number of initally labeled samples
    """
    # Verifying that the local qualit threshold is between 0 and 1
    assert (local_quality_threshold >= 0) and (local_quality_threshold <= 1)

    # Creating a lists with all the samples, annotation status, labels and local qualities
    samples, annotation_status, labels, local_qualities = getSamplesInformation(labeled_samples, unlabeled_samples)
    # Creating two dictionaries which allow to make the correspondance
    # between a sample in one of the previous lists (samples, annotation_status, etc.)
    # and its correspondance in the labeled_samples/unlabeled_samples lists. For
    # intance; if labeled_samples_idxs[i] = j, it means that
    # labeled_samples[labeled_samples_idxs[i]]['Data'] = samples[j] and
    # labeled_samples[labeled_samples_idxs[i]]['Label'] = labels[j], etc.
    labeled_samples_idxs = {j:j for j in range(len(labeled_samples))}
    unlabeled_samples_idxs = {j:j-len(labeled_samples) for j in range(len(labeled_samples), len(labeled_samples)+len(unlabeled_samples))}

    # Verifying if 'True' labels are available for the unlabeled samples (i.e.
    # the label propagation is done for evaluation purposes)
    true_labels_unalabeled_samples_available = True
    for id_sample in range(len(annotation_status)):
        if (annotation_status[id_sample] == False) and (labels[id_sample] is None):
            true_labels_unalabeled_samples_available = False

    # Creating a list to store the new labels (useful in case that the true
    # labels of the unlabeled samples are available)
    new_labels = []
    for id_sample in range(len(annotation_status)):
        if (annotation_status[id_sample] == False):
            new_labels.append(None)
        else:
            new_labels.append(labels[id_sample])

    # Number of classes
    nbClasses = len(np.unique([l for l in labels if l is not None]))

    # Sorting the samples by local qualities
    idxs_samples = np.argsort(local_qualities)[::-1]

    # Computing the K nearest neighbors
    # WARNING: As the query set is the same as the training set (for KNN),
    # the nearest neighbor of each point is the point itself at a distance of zero
    # That is why we search for the K+1 nearest neighbors instead of the K nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=K+1, algorithm='ball_tree').fit(samples)
    distances, indices = neighbors.kneighbors(samples)

    # Doing the labeld propagation
    do_label_propagation = True
    new_annotated_samples = []
    while (do_label_propagation):
        nb_new_samples_per_class = {i:0 for i in range(nbClasses)}
        tmp_newly_labeled_samples = {}
        for id_sample in idxs_samples:
            # We are going to annotate the neighbors of the sample if and only
            # if the sample is annotated and its local quality is good
            if (annotation_status[id_sample]) and (local_qualities[id_sample] > local_quality_threshold):
                tmp_nearest_neighbors = indices[id_sample][1:]
                final_neighbors = [] # This neighbors are the ones that have a local
                # quality greater than the local_quality_threshold AND that are NOT labeled
                for neighbor in tmp_nearest_neighbors:
                    if (local_qualities[neighbor] > local_quality_threshold and annotation_status[neighbor] == False):
                        final_neighbors.append(neighbor)

                # Now that we have the neighbors with a good local quality,
                # we are going to propagate the label of the current sample
                # to its neighbors
                if (len(final_neighbors) > 0):
                    for neighbor in final_neighbors:
                        # Assigning to the neighbor the scores of the current sample
                        if (neighbor not in tmp_newly_labeled_samples):
                            tmp_newly_labeled_samples[neighbor] = [id_sample] # the values
                            # of this dict are the current labeled samples and the
                            # keys are the current unlabeled samples. We do this
                            # because an unlabeled sample takes the label of its
                            # best local quality labeled neighbor
                        else:
                            tmp_newly_labeled_samples[neighbor].append(id_sample)

        for sample_to_label in tmp_newly_labeled_samples:
            labeled_neighbor_with_best_LQ = None
            best_lq = 0
            for labeled_neighbor in tmp_newly_labeled_samples[sample_to_label]:
                if (labeled_neighbor_with_best_LQ is None):
                    labeled_neighbor_with_best_LQ = labeled_neighbor
                    best_lq = local_qualities[labeled_neighbor_with_best_LQ]
                else:
                    if (best_lq < local_qualities[labeled_neighbor]):
                        labeled_neighbor_with_best_LQ = labeled_neighbor
                        best_lq = local_qualities[labeled_neighbor_with_best_LQ]

            # Labeling the sample
            new_labels[sample_to_label] = new_labels[labeled_neighbor_with_best_LQ]

            if (true_labels_unalabeled_samples_available):
                sampleToAdd = unlabeled_samples[unlabeled_samples_idxs[sample_to_label]]
                sampleToAdd['Label'] = new_labels[sample_to_label]
                sampleToAdd['TrueLabel'] = labels[sample_to_label]
                new_annotated_samples.append(sampleToAdd)
            else:
                sampleToAdd = unlabeled_samples[unlabeled_samples_idxs[sample_to_label]]
                sampleToAdd['Label'] = new_labels[sample_to_label]
                new_annotated_samples.append(sampleToAdd)

            nb_new_samples_per_class[new_labels[sample_to_label]] += 1
            annotation_status[sample_to_label] = True

        tot_new_samples = sum(nb_new_samples_per_class.values())
        if (tot_new_samples == 0): # In this case there are no more samples to label
            # Stopping label proapgation
            do_label_propagation = False

            # Getting some useful information
            nb_annotated_samples = len(new_annotated_samples)
            total_number_of_samples = len(samples)
            number_initial_labeled_samples = len(labeled_samples)

            # Counting the total number of correctly labeled samples
            # WARNING: the following lines are only used for evaluation, but if the
            # true label of the unlabeled samples is not known, it is not
            # possible to use them
            accuracy_annotation = None
            if (true_labels_unalabeled_samples_available):
                accuracy_annotation = 0
                for new_annot_sample in new_annotated_samples:
                    if (new_annot_sample['Label'] == new_annot_sample['TrueLabel']):
                        accuracy_annotation += 1
                if (nb_annotated_samples > 0):
                    accuracy_annotation = accuracy_annotation/nb_annotated_samples
                else:
                    accuracy_annotation = 1

            return new_annotated_samples,\
                   accuracy_annotation,\
                   nb_annotated_samples,\
                   total_number_of_samples,\
                   number_initial_labeled_samples

def propagateLabelsLocalQuality_LQKNN_withoutSort(
                            labeled_samples,
                            unlabeled_samples,
                            K=10,
                            local_quality_threshold=0.1
                          ):
    """
        Does label propagation using LQ-KNN without sorting the sampels by local
        quality. It shows that the annotation order in the embedded space
        is important to reduce annotation errors

        Arguments:
        ----------
        labeled_samples: list
            List containing the labeled samples. Each sample (i.e. value of the
            list is a dictionary with THREE keys: 'Data', 'Label', 'LocalQuality')
        unlabeled_samples: dict
            Same as labeled_samples but in this case the 'Label' of each sample
            is None (except if the label is known and the label propagation is
            done for evaluation purposes)
        K: int
            Neighborhood to consider for label propagation
        local_quality_threshold: float
            Threshold used to define the 'good' quality samples. A sample is
            considered as 'good' if its local quality value is greater than
            the defined threshold

        Returns:
        --------
        new_annotated_samples: list
            List of the new annotated samples. Each sample is a dictionary with
            three keys: 'Data', 'Label', 'LocalQuality'
        accuracy_annotation: float or None
            If the true labels of the unlabeled samples are available, the
            annotation accuracy is computed. If not, its value is None
        nb_annotated_samples: int
            Total numbr of labeled samples
        total_number_of_samples: int
            Total number of samples. It may not be equal to
            nb_annotated_samples + labeled_samples because not all the
            unlabeled_samples are always labeled.
        number_initial_labeled_samples int
            Number of initally labeled samples
    """
    # Verifying that the local qualit threshold is between 0 and 1
    assert (local_quality_threshold >= 0) and (local_quality_threshold <= 1)

    # Creating a lists with all the samples, annotation status, labels and local qualities
    samples, annotation_status, labels, local_qualities = getSamplesInformation(labeled_samples, unlabeled_samples)
    # Creating two dictionaries which allow to make the correspondance
    # between a sample in one of the previous lists (samples, annotation_status, etc.)
    # and its correspondance in the labeled_samples/unlabeled_samples lists. For
    # intance; if labeled_samples_idxs[i] = j, it means that
    # labeled_samples[labeled_samples_idxs[i]]['Data'] = samples[j] and
    # labeled_samples[labeled_samples_idxs[i]]['Label'] = labels[j], etc.
    labeled_samples_idxs = {j:j for j in range(len(labeled_samples))}
    unlabeled_samples_idxs = {j:j-len(labeled_samples) for j in range(len(labeled_samples), len(labeled_samples)+len(unlabeled_samples))}

    # Verifying if 'True' labels are available for the unlabeled samples (i.e.
    # the label propagation is done for evaluation purposes)
    true_labels_unalabeled_samples_available = True
    for id_sample in range(len(annotation_status)):
        if (annotation_status[id_sample] == False) and (labels[id_sample] is None):
            true_labels_unalabeled_samples_available = False

    # Creating a list to store the new labels (useful in case that the true
    # labels of the unlabeled samples are available)
    new_labels = []
    for id_sample in range(len(annotation_status)):
        if (annotation_status[id_sample] == False):
            new_labels.append(None)
        else:
            new_labels.append(labels[id_sample])

    # Number of classes
    nbClasses = len(np.unique([l for l in labels if l is not None]))

    # Getting a list with the indices of the samples
    idxs_samples = [i for i in range(len(samples))]

    # Computing the K nearest neighbors
    # WARNING: As the query set is the same as the training set (for KNN),
    # the nearest neighbor of each point is the point itself at a distance of zero
    # That is why we search for the K+1 nearest neighbors instead of the K nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=K+1, algorithm='ball_tree').fit(samples)
    distances, indices = neighbors.kneighbors(samples)

    # Doing the labeld propagation
    do_label_propagation = True
    new_annotated_samples = []
    while (do_label_propagation):
        nb_new_samples_per_class = {i:0 for i in range(nbClasses)}
        tmp_newly_labeled_samples = {}
        for id_sample in idxs_samples:
            # We are going to annotate the neighbors of the sample if and only
            # if the sample is annotated and its local quality is good
            if (annotation_status[id_sample]) and (local_qualities[id_sample] > local_quality_threshold):
                tmp_nearest_neighbors = indices[id_sample][1:]
                final_neighbors = [] # This neighbors are the ones that have a local
                # quality greater than the local_quality_threshold AND that are NOT labeled
                for neighbor in tmp_nearest_neighbors:
                    if (local_qualities[neighbor] > local_quality_threshold and annotation_status[neighbor] == False):
                        final_neighbors.append(neighbor)

                # Now that we have the neighbors with a good local quality,
                # we are going to propagate the label of the current sample
                # to its neighbors
                if (len(final_neighbors) > 0):
                    for neighbor in final_neighbors:
                        # Labeling the sample
                        new_labels[neighbor] = new_labels[id_sample]
                        annotation_status[neighbor] = True
                        nb_new_samples_per_class[new_labels[neighbor]] += 1

                        # Adding the sample to the list of new annotated samples
                        if (true_labels_unalabeled_samples_available):
                            sampleToAdd = unlabeled_samples[unlabeled_samples_idxs[neighbor]]
                            sampleToAdd['Label'] = new_labels[neighbor]
                            sampleToAdd['TrueLabel'] = labels[neighbor]
                            new_annotated_samples.append(sampleToAdd)
                        else:
                            sampleToAdd = unlabeled_samples[unlabeled_samples_idxs[neighbor]]
                            sampleToAdd['Label'] = new_labels[neighbor]
                            new_annotated_samples.append(sampleToAdd)


        tot_new_samples = sum(nb_new_samples_per_class.values())
        if (tot_new_samples == 0): # In this case there are no more samples to label
            # Stopping label proapgation
            do_label_propagation = False

            # Getting some useful information
            nb_annotated_samples = len(new_annotated_samples)
            total_number_of_samples = len(samples)
            number_initial_labeled_samples = len(labeled_samples)

            # Counting the total number of correctly labeled samples
            # WARNING: the following lines are only used for evaluation, but if the
            # true label of the unlabeled samples is not known, it is not
            # possible to use them
            accuracy_annotation = None
            if (true_labels_unalabeled_samples_available):
                accuracy_annotation = 0
                for new_annot_sample in new_annotated_samples:
                    if (new_annot_sample['Label'] == new_annot_sample['TrueLabel']):
                        accuracy_annotation += 1
                if (nb_annotated_samples > 0):
                    accuracy_annotation = accuracy_annotation/nb_annotated_samples
                else:
                    accuracy_annotation = 1

            return new_annotated_samples,\
                   accuracy_annotation,\
                   nb_annotated_samples,\
                   total_number_of_samples,\
                   number_initial_labeled_samples



def propagateLabels_StdKNN(
                            labeled_samples,
                            unlabeled_samples,
                            K=10
                          ):
    """
        Does label propagation using LQ-KNN.

        Arguments:
        ----------
        labeled_samples: list
            List containing the labeled samples. Each sample (i.e. value of the
            list is a dictionary with THREE keys: 'Data', 'Label', 'LocalQuality')
        unlabeled_samples: dict
            Same as labeled_samples but in this case the 'Label' of each sample
            is None (except if the label is known and the label propagation is
            done for evaluation purposes)
        K: int
            Neighborhood to consider for label propagation

        Returns:
        --------
        new_annotated_samples: list
            List of the new annotated samples. Each sample is a dictionary with
            three keys: 'Data', 'Label', 'LocalQuality'
        accuracy_annotation: float or None
            If the true labels of the unlabeled samples are available, the
            annotation accuracy is computed. If not, its value is None
        nb_annotated_samples: int
            Total numbr of labeled samples
        total_number_of_samples: int
            Total number of samples. It may not be equal to
            nb_annotated_samples + labeled_samples because not all the
            unlabeled_samples are always labeled.
        number_initial_labeled_samples int
            Number of initally labeled samples
    """
    # Creating a lists with all the samples, annotation status, labels and local qualities
    samples, annotation_status, labels, local_qualities = getSamplesInformation(labeled_samples, unlabeled_samples)
    # Creating two dictionaries which allow to make the correspondance
    # between a sample in one of the previous lists (samples, annotation_status, etc.)
    # and its correspondance in the labeled_samples/unlabeled_samples lists. For
    # intance; if labeled_samples_idxs[i] = j, it means that
    # labeled_samples[labeled_samples_idxs[i]]['Data'] = samples[j] and
    # labeled_samples[labeled_samples_idxs[i]]['Label'] = labels[j], etc.
    labeled_samples_idxs = {j:j for j in range(len(labeled_samples))}
    unlabeled_samples_idxs = {j:j-len(labeled_samples) for j in range(len(labeled_samples), len(labeled_samples)+len(unlabeled_samples))}

    # Verifying if 'True' labels are available for the unlabeled samples (i.e.
    # the label propagation is done for evaluation purposes)
    true_labels_unalabeled_samples_available = True
    for id_sample in range(len(annotation_status)):
        if (annotation_status[id_sample] == False) and (labels[id_sample] is None):
            true_labels_unalabeled_samples_available = False

    # Creating a list to store the new labels (useful in case that the true
    # labels of the unlabeled samples are available)
    new_labels = []
    for id_sample in range(len(annotation_status)):
        if (annotation_status[id_sample] == False):
            new_labels.append(None)
        else:
            new_labels.append(labels[id_sample])

    # Number of classes
    nbClasses = len(np.unique([l for l in labels if l is not None]))

    # Getting a list with the indices of the samples
    idxs_samples = [i for i in range(len(samples))]

    # Computing the K nearest neighbors
    # WARNING: As the query set is the same as the training set (for KNN),
    # the nearest neighbor of each point is the point itself at a distance of zero
    # That is why we search for the K+1 nearest neighbors instead of the K nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=K+1, algorithm='ball_tree').fit(samples)
    distances, indices = neighbors.kneighbors(samples)

    # Doing the labeld propagation
    do_label_propagation = True
    new_annotated_samples = []
    while (do_label_propagation):
        nb_new_samples_per_class = {i:0 for i in range(nbClasses)}
        tmp_newly_labeled_samples = {}
        for id_sample in idxs_samples:
            # We are going to annotate the neighbors of the sample if and only
            # if the sample is annotated
            if (annotation_status[id_sample]):
                tmp_nearest_neighbors = indices[id_sample][1:]
                final_neighbors = [] # This neighbors are the ones that are NOT labeled
                for neighbor in tmp_nearest_neighbors:
                    if (annotation_status[neighbor] == False):
                        final_neighbors.append(neighbor)

                # Now that we have the neighbors with a good local quality,
                # we are going to propagate the label of the current sample
                # to its neighbors
                if (len(final_neighbors) > 0):
                    for neighbor in final_neighbors:
                        # Labeling the sample
                        new_labels[neighbor] = new_labels[id_sample]
                        annotation_status[neighbor] = True
                        nb_new_samples_per_class[new_labels[neighbor]] += 1

                        # Adding the sample to the list of new annotated samples
                        if (true_labels_unalabeled_samples_available):
                            sampleToAdd = unlabeled_samples[unlabeled_samples_idxs[neighbor]]
                            sampleToAdd['Label'] = new_labels[neighbor]
                            sampleToAdd['TrueLabel'] = labels[neighbor]
                            new_annotated_samples.append(sampleToAdd)
                        else:
                            sampleToAdd = unlabeled_samples[unlabeled_samples_idxs[neighbor]]
                            sampleToAdd['Label'] = new_labels[neighbor]
                            new_annotated_samples.append(sampleToAdd)

        tot_new_samples = sum(nb_new_samples_per_class.values())
        if (tot_new_samples == 0): # In this case there are no more samples to label
            # Stopping label proapgation
            do_label_propagation = False

            # Getting some useful information
            nb_annotated_samples = len(new_annotated_samples)
            total_number_of_samples = len(samples)
            number_initial_labeled_samples = len(labeled_samples)

            # Counting the total number of correctly labeled samples
            # WARNING: the following lines are only used for evaluation, but if the
            # true label of the unlabeled samples is not known, it is not
            # possible to use them
            accuracy_annotation = None
            if (true_labels_unalabeled_samples_available):
                accuracy_annotation = 0
                for new_annot_sample in new_annotated_samples:
                    if (new_annot_sample['Label'] == new_annot_sample['TrueLabel']):
                        accuracy_annotation += 1
                if (nb_annotated_samples > 0):
                    accuracy_annotation = accuracy_annotation/nb_annotated_samples
                else:
                    accuracy_annotation = 1


            return new_annotated_samples,\
                   accuracy_annotation,\
                   nb_annotated_samples,\
                   total_number_of_samples,\
                   number_initial_labeled_samples


def propagateLabels_OPF(
                            labeled_samples,
                            unlabeled_samples,
                        ):
    """
        Does label propagation using Semi-Supervised Optimum-Path Forest.

        Arguments:
        ----------
        labeled_samples: list
            List containing the labeled samples. Each sample (i.e. value of the
            list is a dictionary with THREE keys: 'Data', 'Label', 'LocalQuality')
        unlabeled_samples: dict
            Same as labeled_samples but in this case the 'Label' of each sample
            is None (except if the label is known and the label propagation is
            done for evaluation purposes)

        Returns:
        --------
        new_annotated_samples: list
            List of the new annotated samples. Each sample is a dictionary with
            three keys: 'Data', 'Label', 'LocalQuality'
        accuracy_annotation: float or None
            If the true labels of the unlabeled samples are available, the
            annotation accuracy is computed. If not, its value is None
        nb_annotated_samples: int
            Total numbr of labeled samples
        total_number_of_samples: int
            Total number of samples. It may is equal to
            unlabeled_samples + labeled_samples as all the unlabeled_samples
            are labeled by this method
        number_initial_labeled_samples int
            Number of initally labeled samples
    """
    # Verifying if 'True' labels are available for the unlabeled samples (i.e.
    # the label propagation is done for evaluation purposes)
    true_labels_unalabeled_samples_available = True
    for sample in unlabeled_samples:
        if sample['Label'] is None:
            true_labels_unalabeled_samples_available = False

    # Getting the labeled and unlabeled samples to use with OPF
    X_labeled, Y_labeled = np.array([sample['Data'] for sample in labeled_samples]), np.array([sample['Label'] for sample in labeled_samples])
    X_unlabeled, Y_unlabeled = np.array([sample['Data'] for sample in unlabeled_samples]), np.array([sample['Label'] for sample in unlabeled_samples])

    print("Number of labeled samples: {} ({} %)".format(len(X_labeled), len(X_labeled)/(len(X_labeled)+len(X_unlabeled))))
    print("Number of unlabeled samples: {} ({} %)".format(len(X_unlabeled), len(X_unlabeled)/(len(X_labeled)+len(X_unlabeled))))

    # Doing the propagation
    opf = SemiSupervisedOPF(distance='log_squared_euclidean', pre_computed_distance=None)
    opf.fit(X_labeled, Y_labeled, X_unlabeled)
    predicted_labels_X_unlabeled = opf.predict(X_unlabeled)

    # Some useful information
    number_initial_labeled_samples = len(X_labeled)
    nb_annotated_samples = len(X_unlabeled)
    total_number_of_samples = len(X_unlabeled) + len(X_unlabeled)

    # Computing the annotation accuracy
    accuracy_annotation = None
    if (true_labels_unalabeled_samples_available):
        accuracy_annotation = 0
        nbWronglyLabeledSamples = 0
        for unlabeledSample_ID in range(len(predicted_labels_X_unlabeled)):
            if (predicted_labels_X_unlabeled[unlabeledSample_ID] == Y_unlabeled[unlabeledSample_ID]):
                accuracy_annotation += 1
            else:
                nbWronglyLabeledSamples += 1
        accuracy_annotation = accuracy_annotation/len(Y_unlabeled)
        # print("Annotation Accuracy: {}".format(accuracy_annotation))

    # Creating a list with the new labeled samples
    new_annotated_samples = []
    for id_sample in range(len(X_unlabeled)):
        sampleToAdd = unlabeled_samples[id_sample]
        sampleToAdd['Label'] = predicted_labels_X_unlabeled[id_sample]
        new_annotated_samples.append(sampleToAdd)



    return new_annotated_samples,\
           accuracy_annotation,\
           nb_annotated_samples,\
           total_number_of_samples,\
           number_initial_labeled_samples
