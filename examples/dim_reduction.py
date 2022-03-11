#!/usr/bin/env python3
"""
    Exemple of dimensionality reduction (second step of our proposed method)
"""
import subprocess
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
from utils.download_exps_data import download_dim_red_data, download_organc_mnist
import argparse

def main():
    # =========================================================================#
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("--dataset", default='MNIST', help="Name of the dataset to use", type=str)
    args = vars(ap.parse_args())

    # Points file and target dim
    dataset = args['dataset']

    #=========================================================================#
    #=========================================================================#
    # Downloading the useful data for the experiment
    download_dim_red_data(dataset)

    # Downloading the OrganCMNIST dataset if used
    if (dataset.lower() == 'organcmnist'):
        download_organc_mnist()

    #=========================================================================#
    #=========================================================================#
    # Computing multiple representations with different hyper-parameters
    # for t-SNE
    print("\n==================================================================")
    print("==================Computing multiple projections==================")
    exp_name = 'Example-Dim-Reduction'
    if (dataset.lower() == 'mnist'):
        representations_file = '../models/MNIST_Example_0/CompressedRepresentations/training_representations.pth'
    elif (dataset.lower() == 'organcmnist'):
        representations_file = '../models/OrganCMNIST_Example_0/CompressedRepresentations/training_representations.pth'
    target_dim = 2
    with subprocess.Popen(\
                            [
                                'python',\
                                '../src/tsne_grid_search.py',\
                                '--exp_name',\
                                exp_name,\
                                '--representations_file',\
                                representations_file,\
                                '--target_dim',\
                                str(target_dim)
                            ], stdout=subprocess.PIPE
                         ) as proc:
        for line in proc.stdout:
            line = line.decode("utf-8")
            print(line)


    #=========================================================================#
    #=========================================================================#
    # Selecting the optimal projection using our propsed selection strategy
    # based on the Silhouette Score
    print("\n\n==================================================================")
    print("==================Selecting the optimal projection==================")
    if (dataset.lower() == 'mnist'):
        projections_folder = '../models/MNIST_Example_0/Projections_{}_0/'.format(exp_name)
    elif (dataset.lower() == 'organcmnist'):
        projections_folder = '../models/OrganCMNIST_Example_0/Projections_{}_0/'.format(exp_name)

    with subprocess.Popen(\
                            [
                                'python',\
                                '../src/optimal_projection_selection.py',\
                                '--projections_folder',\
                                projections_folder,\
                                '--percentage_labels_keep',\
                                '0.1'
                            ], stdout=subprocess.PIPE
                         ) as proc:
        for line in proc.stdout:
            line = line.decode("utf-8")
            print(line)



if __name__=="__main__":
    main()
