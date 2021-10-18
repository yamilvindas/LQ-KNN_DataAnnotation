#!/usr/bin/env python3
"""
    Exemple of dimensionality reduction (second step of our proposed method)
"""
import subprocess
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np

def main():
    #=========================================================================#
    #=========================================================================#
    # Computing multiple representations with different hyper-parameters
    # for t-SNE
    print("\n==================================================================")
    print("==================Computing multiple projections==================")
    exp_name = 'Example-Dim-Reduction'
    representations_file = '../models/MNIST_Example_0/CompressedRepresentations/training_representations.pth'
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
    projections_folder = '../models/MNIST_Example_0/Projections_{}_0/'.format(exp_name)
    with subprocess.Popen(\
                            [
                                'python',\
                                '../src/optimal_projection_selection.py',\
                                '--projections_folder',\
                                projections_folder
                            ], stdout=subprocess.PIPE
                         ) as proc:
        for line in proc.stdout:
            line = line.decode("utf-8")
            print(line)



if __name__=="__main__":
    main()
