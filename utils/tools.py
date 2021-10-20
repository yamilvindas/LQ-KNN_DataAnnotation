#!/usr/bin/env python3
"""
    Some useful functions used in the source codes.
"""

import csv
import torch
import numpy as np
import os
import h5py


def computeTestLossAE(model, \
                      test_dataloader, \
                      loss_function,\
                      device=torch.device('cpu')
                     ):
    """
        Compute the test loss of an Autoencoder model

        Parameters
        ----------
        model: torch.Module
            model to compute the confusion matrix
        test_dataloader: torch.utils.data.dataloader.DataLoader
            DataLoader for the test data
        loss_function:
            Loss function to use

        Returns
        -------
        test_loss: float
            Mean test loss over the batches
        test_representations: list
            List of 3D points produced by the model
    """
    model = model.double()
    tmp_losses = []
    model.eval()
    test_representations = {'compressed_representation': [], 'original_image': [], 'label': []}
    with torch.no_grad():
        for sample in test_dataloader:
            points, labels = sample[0], sample[1]
            data = points.double()
            data = data.to(device)
            outputs = model(data)

            # Saving the 3D points of the output to plot them later
            test_representations['compressed_representation'].append(model.compressed_representation.cpu().detach())
            test_representations['label'].append(labels)
            test_representations['original_image'].append(data.cpu().detach())

            # Loss function
            loss = loss_function(outputs, data) # WE DO NOT PUT loss_function(outputs, labels)
            # because we use this to compute the loss of an autoencoder
            loss_data = loss.data.cpu()
            tmp_losses.append(loss_data.numpy())

    loss = np.array(tmp_losses).mean()

    return loss, test_representations

def save_metrics(train_metrics, test_metrics, nameFile):
    """
        Save the train and evaluation metrics in a hdf5 file using h5py

        Parameters:
        -----------
        train_metrics: dict
            Dictionary where the keys are the names of the train  metrics and
            the values are the values of the metrics over the epochs.
        test_metrics: dict
            Same as train_metrics but for the test set
        nameFile: str
            Path of the file where the metrics are going to be saved
    """
    # Saving metrics
    h5f = h5py.File(nameFile, 'a')
    # Training metrics
    for train_metric in train_metrics:
        metric_train_dset = h5f.create_dataset("train/" + train_metric, data=train_metrics[train_metric])

    # Testing metrics
    for test_metric in test_metrics:
        metric_test_dset = h5f.create_dataset("test/" + test_metric, data=test_metrics[test_metric])


    # Closing file
    h5f.close()

    return nameFile


def load_metrics(nameFile='./train_test_metrics.hdf5'):
    """
        Load the train and evaluation metrics in a hdf5 file using h5py

        Parameters:
        -----------
        nameFile: str
            Name of the file to containing the metrics to load

        Returns:
        --------
        train_metrics: dict
            Dictionary where the keys are the names of the train  metrics and
            the values are the values of the metrics over the epochs.
        test_metrics: dict
            Same as train_metrics but for the test set

    """
    # Creating the dict that will contain the train and eval metrics
    train_metrics = {
                        'loss':[]
                    }
    test_metrics = {
                        'loss':[]
                    }

    # Loading metrics
    h5f = h5py.File(nameFile, 'r')
    # Training metrics
    for train_metric in train_metrics:
        train_metrics[train_metric] = h5f['train/' + train_metric][:]
    # Testing metrics
    for test_metric in test_metrics:
        test_metrics[test_metric] = h5f['test/' + test_metric][:]

    # Closing file
    h5f.close()

    return train_metrics, test_metrics
