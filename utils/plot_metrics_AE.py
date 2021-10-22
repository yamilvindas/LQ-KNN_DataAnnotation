#!/usr/bin/env python3
"""
    This code allows to plot the train and test metrics curves obtained
    by training an AutoEncoder model.

    Options:
    --------
    --metrics_file: str
        Path to the metrics .pth file of a trained AE model. This file is
        usually stored in a folder of the form "../models/MNIST_EXP_ID/Model/"
"""

import matplotlib.pyplot as plt
from utils.tools import load_metrics
import argparse

def plotMetrics(train_metrics, test_metrics, metric_name):
    # Figure
    fig = plt.figure()

    # Number of epochs
    epochs = list(range(len(train_metrics[metric_name.lower()])))

    # Plot
    plt.plot(epochs, train_metrics[metric_name.lower()], label="Train "+metric_name)
    plt.plot(epochs, test_metrics[metric_name.lower()], label="Test "+metric_name)
    plt.title(metric_name.upper())
    plt.show()

def main():
    # =========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("--metrics_file", required='True', help="Path to the metrics .pth file of a trained AE model.", type=str)
    args = vars(ap.parse_args())

    # Points file and target dim
    metrics_file = args['metrics_file']

    #==========================================================================#
    # Loading the metrics
    #==========================================================================#
    train_metrics, test_metrics = load_metrics(metrics_file)

    #==========================================================================#
    # Plotting the metrics
    #==========================================================================#
    plotMetrics(train_metrics, test_metrics, 'loss')

if __name__=="__main__":
    main()
