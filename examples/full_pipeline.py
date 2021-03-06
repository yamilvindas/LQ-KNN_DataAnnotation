#!/usr/bin/env python3
"""
    This code applies our proposed method from feature extraction to label
    propagation.

    Options:
    --------
    *--propagation_mode: Mode to propagate labels (propLocalQual,
    classicalProp or OPF-Semi)
    *--local_quality_threshold: Local quality threshold to use if
    propLocalQual mode is used and the variable to study is not the local quality threshold
    *--sorted_qualities: True if wanted to sort the samples by local quality
     when propagating the labels using LQ-KNN
    *--var_to_study: Variable to study (K, percentageLabelsKeep or
    localQualThresh)

"""
import argparse
import os
import subprocess
import json
import shutil

def main():
    #==========================================================================#
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("--propagation_mode", default='propLocalQual', help="", type=str)
    ap.add_argument("--local_quality_threshold", default=0.1, help="", type=float)
    ap.add_argument("--sorted_qualities", default='True', help="", type=str)
    ap.add_argument("--var_to_study", default='K', help="", type=str)


    args = vars(ap.parse_args())

    # Parameters
    propagation_mode = args['propagation_mode']
    local_quality_threshold = args['local_quality_threshold']
    sorted_qualities = args['sorted_qualities']
    if (sorted_qualities.lower() == 'true'):
        sorted_qualities = True
    elif (sorted_qualities.lower() == 'false'):
        sorted_qualities = False
    else:
        raise ValueError("Value {} for option sorted_qualities is not valid".format(sorted_qualities))
    var_to_study = args['var_to_study']

    #==========================================================================#
    #==========================================================================#
    # Feature Extraction
    print("\n==================================================================")
    print("==============Extraction features using an Autoencoder==============")
    with subprocess.Popen(\
                            [
                                'python',\
                                '../src/feature_extraction.py',\
                                '--parameters_file',\
                                '../parameters_files/default_parameters_AE.json'
                            ], stdout=subprocess.PIPE
                         ) as proc:
        for line in proc.stdout:
            line = line.decode("utf-8")
            print(line)
            if ("Model saved at: " in line):
                model_folder_name = line.split(' ')[-1].split('/')[2]

    #==========================================================================#
    #==========================================================================#
    # Computing multiple representations with different hyper-parameters
    # for t-SNE
    print("\n\n==================================================================")
    print("==================Computing multiple projections==================")
    exp_name = 'Example-Dim-Reduction'
    representations_file = '../models/{}/CompressedRepresentations/training_representations.pth'.format(model_folder_name)
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


    #==========================================================================#
    #==========================================================================#
    # Selecting the optimal projection using our propsed selection strategy
    # based on the Silhouette Score
    print("\n\n==================================================================")
    print("==================Selecting the optimal projection==================")
    projections_folder = '../models/{}/Projections_{}_0/'.format(model_folder_name, exp_name)
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

    #==========================================================================#
    #==========================================================================#
    # Computing some projection metrics for the optimal selected projection
    print("\n\n==================================================================")
    print("===Computing projection metrics for the optimal selected projection===")
    # Computing the metrics
    ks, kt = '10', '10'
    quality_lueks = 'True'
    local_quality = 'True'
    with subprocess.Popen(\
                            [
                                'python',\
                                '../src/projection_metrics.py',\
                                '--latent_space_repr',\
                                representations_file,\
                                '--ks',\
                                ks,\
                                '--kt',\
                                kt,\
                                '--quality_lueks',\
                                quality_lueks,\
                                '--local_quality',\
                                local_quality
                            ], stdout=subprocess.PIPE
                         ) as proc:
        for line in proc.stdout:
            line = line.decode("utf-8")
            print(line)

    #==========================================================================#
    #==========================================================================#
    # Doing label propagation
    print("\n\n==================================================================")
    print("====================Doing Label propagation with {}====================".format(propagation_mode))
    exp_ID = 'LabelPropEvaluation'
    with subprocess.Popen(\
                            [
                                'python',\
                                '../examples/label_propagation.py',\
                                '--exp_ID',\
                                exp_ID,\
                                '--projections_folder',\
                                projections_folder,\
                                '--propagation_mode',\
                                propagation_mode,\
                                '--var_to_study',\
                                var_to_study,\
                                '--sorted_qualities',\
                                str(sorted_qualities),\
                                '--local_quality_threshold',\
                                str(local_quality_threshold),\
                                '--ks',\
                                ks,\
                                '--kt',\
                                kt,\
                                '--projection_type_to_use',\
                                'Best'
                            ], stdout=subprocess.PIPE
                         ) as proc:
        for line in proc.stdout:
            line = line.decode("utf-8")
            print(line)






if __name__=='__main__':
    main()
