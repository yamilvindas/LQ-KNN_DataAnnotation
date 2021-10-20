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
from sklearn import metrics
import pickle
import os
import argparse
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

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

def main():
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("--projections_folder", required=True, help="Folder containing different sub-folders corresponding to the projections that we want to study", type=str)
    args = vars(ap.parse_args())

    # Parameters
    projections_folder = args['projections_folder']

    #==========================================================================#
    print("\n====================================================================")
    print("================Starting study of projections of {}================".format(projections_folder))
    # Computing the different metrics
    projectionsList = []
    silhouetteScores = []
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

                # Computing the Silhouette Score of the projection
                silhouette_score = metrics.silhouette_score(points, labels, metric='sqeuclidean')
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

    #==========================================================================#
    # Selecting the best projection
    finalResults = {
                    'BestProjection': projectionsList[maxSilhouetteScoreIndex],
                    'SilhouetteScoreBestProjection': silhouetteScores[maxSilhouetteScoreIndex],
                    'MiddleProjection': projectionsList[midSilhouetteScoreIndex],
                    'SilhouetteScoreMiddleProjection': silhouetteScores[midSilhouetteScoreIndex],
                    'WorstProjection': projectionsList[minSilhouetteScoreIndex],
                    'SilhouetteScoreWorstProjection': silhouetteScores[minSilhouetteScoreIndex]
                    }

    # Creating a file with the results of the best projection
    with open(projections_folder + '/resultsProjectionSelection.json', 'w') as outfile:
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
    plot2DProjetion(best_projection_points, best_projection_labels, best_projection_silhouette_score)

    # Middle projection
    middle_projection_folder = finalResults['MiddleProjection']
    middle_projection_silhouette_score = finalResults['SilhouetteScoreMiddleProjection']
    with open(middle_projection_folder + '/representations_0.pth', "rb") as fp:   # Unpickling
        middle_projection_points = pickle.load(fp)
    with open(middle_projection_folder + '/labels_0.pth', "rb") as fp:   # Unpickling
        middle_projection_labels = pickle.load(fp)
    plot2DProjetion(middle_projection_points, middle_projection_labels, middle_projection_silhouette_score)

    # Worst projection
    worst_projection_folder = finalResults['WorstProjection']
    worst_projection_silhouette_score = finalResults['SilhouetteScoreWorstProjection']
    with open(worst_projection_folder + '/representations_0.pth', "rb") as fp:   # Unpickling
        worst_projection_points = pickle.load(fp)
    with open(worst_projection_folder + '/labels_0.pth', "rb") as fp:   # Unpickling
        worst_projection_labels = pickle.load(fp)
    plot2DProjetion(worst_projection_points, worst_projection_labels, worst_projection_silhouette_score)

    #==========================================================================#
    # Ending the code
    print("================Ending study of projections of {}================".format(projections_folder))
    print("====================================================================\n\n")


if (__name__=="__main__"):
    main()
