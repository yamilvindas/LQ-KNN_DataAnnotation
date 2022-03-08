#!/usr/bin/env python3
"""
    Plot the results of a classification experiment using a semi-automatically
    labeled dataset

    Options:
    ----------
    --classif_res_folder: str
        Path to the folder containing the classfication results. This folder
        is usually obtained using the code example/label_propagation_with_classification.py
"""
import os
import pickle
import argparse
import itertools
import matplotlib.pyplot as plt
from utils.download_exps_data import download_label_propagation_results_classification_data

def plotClassifResults(classif_res_folder):
    """
        Plot the classification results on the files located in classif_res_folder
        using boxplots.

        Arguments:
        ----------
        classif_res_folder: str
            Path to the folder containing the classfication results. This folder
            is usually obtained using the code
            example/label_propagation_with_classification.py
    """
    # Plotting boxplots parameters
    box_plot_offset_step = 1
    boxprops = dict(linestyle='-', linewidth=2.5)
    whiskerprops = dict(linestyle='-', linewidth=2.5)
    flierprops = dict(marker='o', markersize=12)
    medianprops = dict(linestyle='-', linewidth=2.0)
    meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick')
    axisTickSize = 30
    fig = plt.figure()
    plt.grid()
    box_plot_offset = 0

    # Colors for the plots
    c = ['xkcd:red',\
         'xkcd:dark red',\
         'xkcd:blue',\
         'xkcd:dark blue',\
         'xkcd:green',\
         'xkcd:dark green',\
         'xkcd:purple',\
         'xkcd:dark purple',\
         'xkcd:brown',\
         'xkcd:dark brown',\
        ]
    c = itertools.cycle(c)


    # Getting the list of results
    results_list = {}
    for file_name in os.listdir(classif_res_folder):
        if (os.path.isfile(classif_res_folder+'/'+file_name)):
            results_file = classif_res_folder + '/' + file_name
            try:
                # Loading the data
                with open(results_file, "rb") as fp:
                    results = pickle.load(fp)
                # Getting the results at the last epoch
                last_epoch_accs = []
                for repetitionNb in range(len(results)): # For data obtained with the MNIST Git
                    last_epoch_accs.append(results[repetitionNb]['TestAcc'][-1])
                results_list[file_name] = last_epoch_accs
            except:
                pass

    # Creating the plotting boxes
    for experiment in sorted(results_list.keys()):
        # For the accuracies
        plt.xlabel("Experiment")
        plt.ylabel("Accuracy")
        plt.xticks(fontsize=axisTickSize)
        plt.yticks(fontsize=axisTickSize)
        data_bp = results_list[experiment]
        experiment_name = '.'.join(experiment.split('.')[:-1])
        experiment_name = '\n'.join(experiment_name.split('_')[:-1])
        bpl = plt.boxplot(
                            data_bp,\
                            positions=[box_plot_offset],\
                            sym='o',\
                            widths=0.2,\
                            labels=[experiment_name],\
                            showmeans=True,\
                            showfliers=True,\
                            notch=True,\
                            boxprops=boxprops,\
                            flierprops=flierprops,\
                            whiskerprops=whiskerprops,\
                            medianprops=medianprops,\
                            meanprops=meanprops
                        )

        color = next(c)
        plt.setp(bpl['boxes'], color=color)
        plt.setp(bpl['whiskers'], color=color)
        plt.setp(bpl['caps'], color=color)
        plt.setp(bpl['medians'], color=color)

        box_plot_offset += box_plot_offset_step

    # Legend
    plt.legend()
    plt.grid(True)
    # Show
    plt.show()


def main():
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    default_classif_res_folder = '../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/ClassificationResults/'
    ap.add_argument('--classif_res_folder', default=default_classif_res_folder, help="Path to the folder containing the classfication results (obtained using the code example/label_propagation_with_classification.py)", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    classif_res_folder = args['classif_res_folder']

    #==========================================================================#
    # If the default parameters are used, we area going to download the
    # useful data if it has not been done already
    if (classif_res_folder == default_classif_res_folder):
        download_label_propagation_results_classification_data()

    #==========================================================================#
    # Matplib fontsize parameter
    plt.rcParams.update({'font.size': 20})

    #==========================================================================#
    # Plotting the features
    plotClassifResults(classif_res_folder)



if __name__=='__main__':
    main()
