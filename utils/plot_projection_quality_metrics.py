#!/usr/bin/env python3
"""
    This code allows to plot the different projection metrics obtained using
    the code projection_metrics.property

    Options:
    --------
        *--projections_file_folder: str
            Path to the folder containing the representations_0.pth file
            corresponding to the final embedded representation of the original
            samples (in general 2D points)
        *--projection_metric_file: str
            Path to the file contraining the projection metric
"""

import os
import pickle
import argparse
import matplotlib.pyplot as plt
from src.projection_metrics import loadHighAndLowData

def plotLocalQuality(projections_file_folder, local_quality_file_path):
    """
        Plots the local quality for a set of 2D projected points

        Arguments:
        ----------
        projections_folder: str
            Path to the folder containing the representations_0.pth file
            corresponding to the final embedded representation of the original
            samples (in general 2D points)
        local_quality_file_path: str
            Path to the file containing the local quality of the embedded
            representations of the original samples
    """
    # Loading the data
    _, lowDimData = loadHighAndLowData(projections_file_folder)


    # Loading the local qualities
    with open(local_quality_file_path, "rb") as fp:   # Unpickling
        local_qualities = pickle.load(fp)
        # NORMALIZING THE LOCAL QUALITIES
        maxLocalQuals = max(local_qualities)
        local_qualities = [localQual/maxLocalQuals for localQual in local_qualities]

    # Plotting the embedded points with the color
    plt.rcParams['font.size'] = '50' # Changing the fontsize
    fig = plt.figure()
    plt.scatter(lowDimData[:,0], lowDimData[:,1], c=local_qualities, marker='*', s=200)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plotLueksQuality(lueks_quality_file):
    """
        Plots the projection quality based on the definition of Luels et al. (2011)
    """
    # Loading the data
    with open(lueks_quality_file, "rb") as fp:   # Unpickling
        qualityLueks = pickle.load(fp)

    qualityLueksVals = qualityLueks['qualityVals']
    k_s_vals = qualityLueks['k_s_vals']
    k_t_vals = qualityLueks['k_t_vals']

    # Plotting the vals of the new quality using a colored matrix
    fig, ax = plt.subplots()
    psm = ax.pcolormesh(qualityLueksVals, cmap='gist_gray')
    fig.colorbar(psm, ax=ax)
    plt.xlabel("k_t (error tolerance)")
    plt.ylabel("k_s (rank significance)")
    plt.xticks([i for i in range(len(k_s_vals))], k_s_vals)
    plt.yticks([i for i in range(len(k_t_vals))], k_t_vals)
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()

def plotQualityMetrics(metric_file_path):
    """
        Allows to plot other metrics such as Trustworthiness, Continuity, LCMC
        and Quality
    """
    # Loading the data
    with open(metric_file_path, "rb") as fp:   # Unpickling
        metric_vals = pickle.load(fp)
    K_vals = list(range(len(metric_vals)))

    # Plotting the vals of the new quality using a colored matrix
    plt.plot(K_vals, metric_vals)
    plt.xlabel("K")
    plt.ylabel("Metric value")
    plt.title( '.'.join(metric_file_path.split('/')[-1].split('.')[:-1]) )
    plt.show()


def main():
    # =========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("--projections_file_folder", required=False, help="Path to the file containing the projections from which we want to plot the metrics. Used when plotting the local qualities", type=str)
    ap.add_argument("--projection_metric_file", required=False, help="Path to the file contraining the projection metric", type=str)

    args = vars(ap.parse_args())

    # Parameters
    projections_file_folder = args['projections_file_folder']
    projection_metric_file = args['projection_metric_file']


    #==========================================================================#
    if ("local" in projection_metric_file.lower()) and ("quality" in projection_metric_file.lower()):
        if (projections_file_folder is None):
            raise ValueError("A file with the 2D projections of the original data is needed when plotting the local quality")
        else:
            plotLocalQuality(projections_file_folder, projection_metric_file)
    elif ("lueks" in projection_metric_file.lower()) and ("quality" in projection_metric_file.lower()):
        plotLueksQuality(projection_metric_file)
    else:
        plotQualityMetrics(projection_metric_file)


if __name__=="__main__":
    main()
