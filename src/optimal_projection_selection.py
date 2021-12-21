#!/usr/bin/env python3
"""
    This codes select the best projection (obtained by a dimensionality reduction
    technique) based on the Silhouette Score.

    Options:
    --projections_folder: str
        Mandatory, it corresponds to a folder containing different sub-folders
        corresponding to the projections that we want to study

    It generates a file ../models/MNIST_EXP_ID/Projections/resultsProjectionSelection.json
    containing the best, middle and worst projections with their respectives
    Silhouettes Scores
"""
import os
import json
import pickle
import argparse

import random
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

from tqdm import tqdm

def plot2DProjetion(points, labels, silhouette_score):
    # This function only works for 2D points
    assert (len(points[0]) == 2)

    # Transforming the points and labels in numpy arrays if not done
    points, labels = np.array(points), np.array(labels)


    # Getting the names of the labels
    labels_names = np.unique(labels)

    # Plotting the points
    x_vals, y_vals = points[:, 0], points[:, 1]
    for label_name in labels_names:
        idxs_samples = np.where(labels == label_name)# Samples having as label 'label_name'

        plt.scatter(x_vals[idxs_samples], y_vals[idxs_samples], label=str(label_name))
    plt.title("Silhouette score of {}".format(silhouette_score))
    plt.legend(fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def select_labeled_samples(labels_list, percentage_labels_keep):
    """
        From a set of labeled samples, it selects a sub-set that is going to
        be considered as labeled, the rest of the samples are going to be
        considered as unlabeled.

        Arguments:
        ----------
        labels_list: list
            List containing the labels. This is necessary to guarantee that
            we are going to have approximately the same quantity of labeled
            samples per class
        percentage_labels_keep: float
            Percentage of labels to keep

        Returns:
        --------
        labeled_samples_idx: list
            List of the indices of the samples considered as labeled. For instance,
            if sample i is considered as labeled then the value "i" is going to
            be in labeled_samples_idx
        unlabeled_samples_idx: list
            Same as labeled_samples_idx but for the unlabeled samples
    """
    # Initiliazing the lists of labeled and unlabeled samples
    labeled_samples_idx, unlabeled_samples_idx = [], []

    # Creating a list of the annotation status of the samples
    # Here we are going to artificially keep only a percentage of the annotated samples
    # for each class
    nb_samples = len(labels_list)

    # First let's count the number of annotated classes per class
    unique_labels = set(labels_list)
    tot_nb_samples_per_class = {label_val:0 for label_val in unique_labels}
    list_idx_per_class = {label_val:[] for label_val in unique_labels}
    for idSample in range(nb_samples):
        label = labels_list[idSample]
        tot_nb_samples_per_class[label] += 1
        list_idx_per_class[label].append(idSample)

    # Generating the list of annotated status of the samples
    # Labeled samples
    nb_samples_to_label_per_class = {label_val:int(percentage_labels_keep*tot_nb_samples_per_class[label_val]) for label_val in unique_labels}
    for class_val in list_idx_per_class:
        idxs_labels_keep_per_class = random.sample(list_idx_per_class[class_val], nb_samples_to_label_per_class[class_val])
        for sample_id in idxs_labels_keep_per_class: # Labeled samples
            labeled_samples_idx.append(sample_id)
    # Unlabeled samples
    for sample_id in range(nb_samples):
        if (sample_id not in labeled_samples_idx):
            unlabeled_samples_idx.append(sample_id)

    print("There is a total of {} samples labeled over {} samples (proportion of {})".format(len(labeled_samples_idx), len(labeled_samples_idx)+len(unlabeled_samples_idx), len(labeled_samples_idx)/(len(labeled_samples_idx)+len(unlabeled_samples_idx))))

    return labeled_samples_idx, unlabeled_samples_idx

def get_new_labels(labels, labeled_idxs, unlabeled_idxs):
    """
        Allows to get a new list of labels where the labels of indices in
        labeled_idxs are true labels and labels in unlabeled_idxs are
        -1 (i.e. unlabeled)
    """
    new_labels = [-1 for i in range(len(labels))]
    for labeled_idx in labeled_idxs:
        new_labels[labeled_idx] = labels[labeled_idx]
    return new_labels


def select_optimal_projection(projections_folder, percentage_labels_keep):
    """
        Selects the optimal projection among a set of projections based on
        the Silhouette Score of the available labeled samples (the unlabeled
        samples are not taken into account for this)

        Arguments:
        ----------
        projections_folder: str
            Folder containing different sub-folders corresponding to the projections that we want to study
        percentage_labels_keep: float
            Percentage of labels to keep (the other samples are going to be considered from now on as unlabeled)

        Returns:
        --------
        finalResults: dict
            Dictionary with the final results of the selection. It has the
            following keys:
                -'BestProjection'
                -'SilhouetteScoreBestProjection'
                -'MiddleProjection'
                -'SilhouetteScoreMiddleProjection'
                -'WorstProjection'
                -'SilhouetteScoreWorstProjection'
                -'LabeledSamplesIdx'
                -'UnlabeledSamplesIdx'
    """
    #==========================================================================#
    print("\n====================================================================")
    print("================Starting study of projections of {}================".format(projections_folder))
    # Computing the different metrics
    projectionsList = []
    silhouetteScores = []
    areLabeledSamplesSelected = False
    for folderName in tqdm(os.listdir(projections_folder)):
        if ( os.path.isdir(projections_folder + '/' + folderName) and ('embedded' in folderName.lower()) ):
            if (len(os.listdir(projections_folder + '/' + folderName)) > 0):
                # Saving the name of the projection to be able to get it when computing the min and max to
                # get the worst and best projections
                projectionsList.append(projections_folder + '/' + folderName)

                # Getting the 2D points coordinates file
                pointsFile = projections_folder + '/{}/representations_0.pth'.format(folderName)
                with open(pointsFile, "rb") as fp:   # Unpickling
                    points = pickle.load(fp)

                # Getting the labels
                labelsFile = projections_folder + '/{}/labels_0.pth'.format(folderName)
                with open(labelsFile, "rb") as fp:   # Unpickling
                    labels = pickle.load(fp)

                # We are going to keep only a few labels, the other samples are going
                # to be considered as unlabeled
                if (not areLabeledSamplesSelected):
                    # Selecting the labeled and unlabeled samples
                    labeled_samples_idx, unlabeled_samples_idx = select_labeled_samples(labels, percentage_labels_keep)

                    # Saving the selection status to avoid selecting different
                    # labeled samples for each projection
                    areLabeledSamplesSelected = True

                new_points = [points[i] for i in labeled_samples_idx]
                new_labels = [labels[i] for i in labeled_samples_idx]

                # Computing the Silhouette Score of the projection
                silhouette_score = metrics.silhouette_score(new_points, new_labels, metric='sqeuclidean')
                silhouetteScores.append(silhouette_score)


    #==========================================================================#
    # Determining the best, middle and worst projections
    # Sorting the scores of each metric and getting the indexes (to be able to get the associated projection)
    sortedSilhouetteScoresIndexes = sorted(range(len(silhouetteScores)), key=lambda k: silhouetteScores[k])

    # Getting the projections of the best and worst projections for each metric
    # Silhouette Scores: the higher the better
    minSilhouetteScoreIndex = sortedSilhouetteScoresIndexes[0]
    midSilhouetteScoreIndex = sortedSilhouetteScoresIndexes[len(sortedSilhouetteScoresIndexes)//2]
    maxSilhouetteScoreIndex = sortedSilhouetteScoresIndexes[-1]
    print("\n\nBest projection according to the Silhouette Score: {} (Silhouette score of {})".format(projectionsList[maxSilhouetteScoreIndex], silhouetteScores[maxSilhouetteScoreIndex]))
    print("Middle projection according to the Silhouette Score: {} (Silhouette score of {})".format(projectionsList[midSilhouetteScoreIndex], silhouetteScores[midSilhouetteScoreIndex]))
    print("Worst projection according to the Silhouette Score: {} (Silhouette score of {})\n\n".format(projectionsList[minSilhouetteScoreIndex], silhouetteScores[minSilhouetteScoreIndex]))

    finalResults = {
                    'BestProjection': projectionsList[maxSilhouetteScoreIndex],
                    'SilhouetteScoreBestProjection': silhouetteScores[maxSilhouetteScoreIndex],
                    'MiddleProjection': projectionsList[midSilhouetteScoreIndex],
                    'SilhouetteScoreMiddleProjection': silhouetteScores[midSilhouetteScoreIndex],
                    'WorstProjection': projectionsList[minSilhouetteScoreIndex],
                    'SilhouetteScoreWorstProjection': silhouetteScores[minSilhouetteScoreIndex],
                    'LabeledSamplesIdx': labeled_samples_idx,
                    'UnlabeledSamplesIdx': unlabeled_samples_idx
                    }

    return finalResults

def main():
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("--projections_folder", required=True, help="Folder containing different sub-folders corresponding to the projections that we want to study", type=str)
    ap.add_argument("--percentage_labels_keep", required=True, help="Percentage of labels to keep (the other samples are going to be considered from now on as unlabeled)", type=float)
    args = vars(ap.parse_args())

    # Parameters
    projections_folder = args['projections_folder']
    percentage_labels_keep = args['percentage_labels_keep']


    #==========================================================================#
    # Selecting the optimal projection
    finalResults = select_optimal_projection(projections_folder, percentage_labels_keep)

    # Creating a file with the results of the best projection
    inc = 0
    while (os.path.isfile(projections_folder + '/resultsProjectionSelection_{}-labeledSamples_{}.json'.format(percentage_labels_keep, inc))):
        inc += 1
    results_file = projections_folder + '/resultsProjectionSelection_{}-labeledSamples_{}.json'.format(percentage_labels_keep, inc)
    with open(results_file, 'w') as outfile:
        json.dump(finalResults, outfile)


    #==========================================================================#
    # Plotting the best, middle and worst projections
    # Best projection
    best_projection_folder = finalResults['BestProjection']
    best_projection_silhouette_score = finalResults['SilhouetteScoreBestProjection']
    with open(best_projection_folder + '/representations_0.pth', "rb") as fp:   # Unpickling
        best_projection_points = pickle.load(fp)
    with open(best_projection_folder + '/labels_0.pth', "rb") as fp:   # Unpickling
        best_projection_labels = pickle.load(fp)
    best_projection_labels = get_new_labels(best_projection_labels, finalResults['LabeledSamplesIdx'], finalResults['UnlabeledSamplesIdx'])
    plot2DProjetion(best_projection_points, best_projection_labels, best_projection_silhouette_score)

    # Middle projection
    middle_projection_folder = finalResults['MiddleProjection']
    middle_projection_silhouette_score = finalResults['SilhouetteScoreMiddleProjection']
    with open(middle_projection_folder + '/representations_0.pth', "rb") as fp:   # Unpickling
        middle_projection_points = pickle.load(fp)
    with open(middle_projection_folder + '/labels_0.pth', "rb") as fp:   # Unpickling
        middle_projection_labels = pickle.load(fp)
    middle_projection_labels = get_new_labels(middle_projection_labels, finalResults['LabeledSamplesIdx'], finalResults['UnlabeledSamplesIdx'])
    plot2DProjetion(middle_projection_points, middle_projection_labels, middle_projection_silhouette_score)

    # Worst projection
    worst_projection_folder = finalResults['WorstProjection']
    worst_projection_silhouette_score = finalResults['SilhouetteScoreWorstProjection']
    with open(worst_projection_folder + '/representations_0.pth', "rb") as fp:   # Unpickling
        worst_projection_points = pickle.load(fp)
    with open(worst_projection_folder + '/labels_0.pth', "rb") as fp:   # Unpickling
        worst_projection_labels = pickle.load(fp)
    worst_projection_labels = get_new_labels(worst_projection_labels, finalResults['LabeledSamplesIdx'], finalResults['UnlabeledSamplesIdx'])
    plot2DProjetion(worst_projection_points, worst_projection_labels, worst_projection_silhouette_score)

    #==========================================================================#
    # Ending the code
    print("================Ending study of projections of {}================".format(projections_folder))
    print("====================================================================\n\n")


if (__name__=="__main__"):
    main()
