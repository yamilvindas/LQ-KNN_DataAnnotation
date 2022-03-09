#!/usr/bin/env python3
"""
    This code implements the classes needed to load the medMNIST data and
    create a pytorch dataset.
"""
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import h5py

class MedMNIST(Dataset):
    """MedMNIST dataset"""
    def __init__(self, data):
        """
        Args:
            data (dict): the data structured in a dict where the key is the id of the sample
            and the value is another dict with two keys 'ImagePath' and 'Label'

        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        sample = self.data[idx]
        label = torch.tensor(sample['Label'])

        # Getting the image
        sample_data = np.asarray(Image.open(sample['SamplePath']))/255.
        if (len(sample_data.shape) == 3):
            sample_data = np.moveaxis(sample_data, 2, 0) # Because in pytorch the channel has to be in the first position
        else:
            sample_data = sample_data.reshape((1, sample_data.shape[0], sample_data.shape[1]))
        data = torch.tensor(sample_data)
        data = data.type("torch.FloatTensor") # To avoid problems with types

        # Returning the final sample
        return data, label


def loadFromHDF5(hdf5_file_path):
    """
        Create two dictionaries containing the training and test datasets

        Parameters:
        -----------
        hdf5_file_path: str
            Path to an hdf5 file containig the structure of the data to use

        Returns:
        --------
        train_data: dict
            Dictionary where the key is the id of a sample and the value is the
            path to the image sample
        test_data: dict
            Dictionary where the key is the id of a sample and the value is the
            path to the image sample
    """
    # Creating the dictionaries for the dataset
    train_data, val_data, test_data = {}, {}, {}

    # Loading the data
    h5f = h5py.File(hdf5_file_path, 'r')
    for dataset in h5f:
        # print("Dataset: ", dataset)
        for data_split in h5f[dataset]:
            # print("Data split: ", data_split)
            for sampleID in h5f[dataset][data_split]:
                # print("sampleID: ", sampleID)
                if (data_split == 'train'):
                    if (int(sampleID) in train_data):
                        print("In the training split, the sample of ID {} has already been stored".format(sampleID))
                    else:
                        train_data[int(sampleID)] = {
                                                        'Label': h5f[dataset][data_split][sampleID].attrs['Label'],\
                                                        'SamplePath': h5f[dataset][data_split][sampleID].attrs['SamplePath']
                                                    }
                elif (data_split == 'val'):
                    if (int(sampleID) in val_data):
                        print("In the validation split, the sample of ID {} has already been stored".format(sampleID))
                    else:
                        val_data[int(sampleID)] = {
                                                        'Label': h5f[dataset][data_split][sampleID].attrs['Label'],\
                                                        'SamplePath': h5f[dataset][data_split][sampleID].attrs['SamplePath']
                                                  }
                elif (data_split == 'test'):
                    if (int(sampleID) in test_data):
                        print("In the testing split, the sample of ID {} has already been stored".format(sampleID))
                    else:
                        test_data[int(sampleID)] = {
                                                        'Label': h5f[dataset][data_split][sampleID].attrs['Label'],\
                                                        'SamplePath': h5f[dataset][data_split][sampleID].attrs['SamplePath']
                                                    }
                else:
                    raise ValueError("Data split {} not recognized".format(data_split))

    # Closing file
    h5f.close()

    # # Verifying the number of samples per split
    # print("Number of samples for training: ", len(train_data))
    # print("Number of samples for validation: ", len(val_data))
    # print("Number of samples for testing: ", len(test_data))

    return train_data, val_data, test_data

def mix_data_splits(data_split_1, data_split_2):
    """
        Allows to mix two data splits obtained by loadFromHDF5 into one unique
        data split. It can be useful if we want to mix the training and validation
        splits in one split.

        Arguments:
        ----------
        data_split_1: dict
            Dictionary where the key is the id of a sample and the value is the
            path to the image sample
        data_split_2: dict
            Dictionary where the key is the id of a sample and the value is the
            path to the image sample

        Return:
        -------
        mixed_data_split: dict
            Dictionary mixing the samples for data_split_1 and data_split_2.
            The key is the id of a sample and the value is the
            path to the image sample.
    """
    mixed_data_split = {}
    sample_id = 0
    for sample_id_data_split_1 in data_split_1:
        mixed_data_split[sample_id] = data_split_1[sample_id_data_split_1]
        sample_id += 1
    for sample_id_data_split_2 in data_split_2:
        mixed_data_split[sample_id] = data_split_2[sample_id_data_split_2]
        sample_id += 1
    return mixed_data_split
