#!/usr/bin/env python3
"""
    Plot the annotation accuracy and the number of labeled samples obtained
    when evaluating a label propagation method the code
    examples/evaluation_label_propagation.py

    Options:
    --------
        *--label_prop_results_folder: str
            Path to a label propagation results folder (obtained using the code
            examples/evaluation_label_propagation.py)
        *--var_to_study: str
            Studied variable which indicates which results have to be plotted
"""
import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.download_exps_data import download_label_propagation_results_data

def plotAnnotationAccuracy(label_prop_results_folder, var_to_study):
    """
        Plots the annotation accuracy with respect to the studied variable
    """
    # Loading the results
    results = {}
    for file_name in os.listdir(label_prop_results_folder):
        if (os.path.isfile(label_prop_results_folder + '/' + file_name)):
            if ("var-to-study-"+var_to_study.lower() in file_name.lower()):
                with open(label_prop_results_folder + '/' + file_name, 'rb') as fp:
                    results['.'.join(file_name.split('.')[:-1])] = pickle.load(fp)

    # Plotting the annotation accuracy
    for result_name in results:
        plot_name = result_name.split('_')[0]
        # Values of the x-axis
        if ('opf-semi' in result_name.lower()):
            x_vals = [1]
        else:
            x_vals = list(results[result_name]['accs'].keys())
        x_vals.sort()

        # Values of the y-axis
        y_mean, y_std = [], []
        for x in x_vals:
            if ('opf-semi' in result_name.lower()):
                y_mean.append(np.mean(results[result_name]['accs']))
                y_std.append(np.std(results[result_name]['accs']))
            else:
                y_mean.append(np.mean(results[result_name]['accs'][x]))
                y_std.append(np.std(results[result_name]['accs'][x]))

        # Plotting the error bars
        plt.errorbar(x_vals, y_mean, yerr=y_std, label=plot_name, fmt='*')
        plt.xlabel("Annotation accuracy")
        plt.ylabel(var_to_study)
    plt.legend()
    plt.show()

    # Plotting the number of labeled samples
    for result_name in results:
        plot_name = result_name.split('_')[0]
        # Values of the x-axis
        if ('opf-semi' in result_name.lower()):
            x_vals = [1]
        else:
            x_vals = list(results[result_name]['nbsAnnotatedSamples'].keys())
            x_vals.sort()

        # Values of the y-axis
        y_mean, y_std = [], []
        for x in x_vals:
            if ('opf-semi' in result_name.lower()):
                y_mean.append(np.mean(results[result_name]['nbsAnnotatedSamples']))
                y_std.append(np.std(results[result_name]['nbsAnnotatedSamples']))
            else:
                y_mean.append(np.mean(results[result_name]['nbsAnnotatedSamples'][x]))
                y_std.append(np.std(results[result_name]['nbsAnnotatedSamples'][x]))

        # Plotting the error bars
        plt.errorbar(x_vals, y_mean, yerr=y_std, label=plot_name, fmt='*')
        plt.xlabel("Number of labeled samples")
        plt.ylabel(var_to_study)
    plt.legend()
    plt.show()

def main():
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    default_var_to_study = 'K'
    ap.add_argument('--label_prop_results_folder', default=None, help="Path to a label propagation results folder (obtained using the code examples/evaluation_label_propagation.py)", type=str)
    ap.add_argument('--var_to_study', default=default_var_to_study, help="Studied variable which indicates which results have to be plotted. Four choices: K, percentageLabelsKeep, localQualThresh and None", type=str)
    ap.add_argument('--default_dataset', default='MNIST', help="Default dataset to plot the results (it uses precomputed metics). Two options: MNIST and OrganCMNIST. Only used if --label_prop_results_folder is None", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    label_prop_results_folder = args['label_prop_results_folder']
    var_to_study = args['var_to_study']
    default_dataset = args['default_dataset']

    #==========================================================================#
    # If the default parameters are used, we area going to download the
    # useful data if it has not been done already
    if (label_prop_results_folder is None):
        if (default_dataset.lower() == 'mnist'):
            download_label_propagation_results_data('MNIST')
            label_prop_results_folder = '../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/LabelPropResults/'
        elif (default_dataset.lower() == 'organcmnist'):
            download_label_propagation_results_data('OrganCMNIST')
            label_prop_results_folder = '../models/OrganCMNIST_Example_0/Projections_Example-Dim-Reduction_0/LabelPropResults/'

    #==========================================================================#
    # Matplotlib fontsize parameter
    plt.rcParams.update({'font.size': 20})

    #==========================================================================#)
    # Plotting the results
    plotAnnotationAccuracy(label_prop_results_folder, var_to_study)


if __name__=='__main__':
    main()
