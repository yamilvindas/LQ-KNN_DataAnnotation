#!/usr/bin/env python3
"""
    This code does a dimensionality reduction of the input data using a
    Convolutional Autoencoder
    It has one option:
        * --parameters_file: A json file containing the parameters for training the
        autoencoder (name of the experiment, number of epochs, batch size, learning
        rate, weight decay and loss function). If this option is not given,
        we will use the default file located in ../parameters_files/

    This code produces different files:
        * ../models/DATASET-NAME_EXP_ID/Model/model.pth: A pth file containing the parameters
        of the trained autoencoder.
        * ../models/DATASET-NAME_EXP_ID/Model/metrics.hdf5: An hdf5 file containing
        the metrics of the training and testing of the model
        * ../modelsDATASET-NAMET_EXP_ID/CompressedRepresentations/training_representations.pth:
        A pth file containing different information from the different compressed
        training points. When loaded into python it is a list where each element
        corresponds to a dict representing the points over the different epochs.
        The keys of each element (which are dicts) are:
            - 'compressed_representation': Compressed Representation (in terms
            of dimensionality) of the original data sample
            - 'label': Class (or scores) of the reduced dim sample
        * ../models/DATASET-NAME_EXP_ID/CompressedRepresentations/testing_representations_ID.txt:
        Same as before but for the testing data.
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from ae_model import MnistConvolutionalAE, weights_init
import torchvision
from torchsummary import summary
import argparse
import json
import pickle
import numpy as np
import os
from random import randint
import matplotlib.pyplot as plt
from utils.tools import save_metrics, computeTestLossAE
from utils.plot_metrics_AE import plotMetrics
from src.medMNIST import loadFromHDF5, mix_data_splits, MedMNIST

torch.multiprocessing.set_sharing_strategy('file_system') # To allow many files to open

class my_expe_class(object):
    """
        Class corresponding to the experiment of feature extraction using a
        convolutional autoencoder on the MNIST dataset.

        Attributes:
        -----------
        exp_id: str
            Name of the experiment.
        epochs: int
            Number of epochs to use for training.
        batch_size: int
            Batch size to use for training.
        lr: float
            Learning rate to use for training.
        weight_decay: float
            Weigh decay to use for training.
        loss_function:
            Loss function to train the model. Two options: 'MSE_Loss' or 'BCE_Loss'.
        device: str
            Device to train the model. Two options: 'CPU' or 'GPU'.
        latent_space_size: int
            Size of the latent space of the autoencoder. In other words, size
            of the compressed representation of the original input.
        nbSamplesUseMNIST: int
            Number of training samples of the MNIST dataset to use in order
            to train the Convolutional Autoencoder
        dataset_name: str
            Name of the dataset to use for feature extraction. Two options
            are possible: MNIST and OrganCMNIST.
    """
    def __init__(
                    self, exp_id, epochs, batch_size, lr, weight_decay, loss_function,\
                    device='cpu', model_type="MnistConvolutionalAE",\
                    latent_space_size=32, nbSamplesUseMNIST=15000,\
                    dataset_name='MNIST'
                ):

        print("\nInitializing experiment")
        print("=======> Beginning Loading the data <=======\n")
        # experiment ID
        self.exp_id = exp_id

        # Number of samples to use for MNIST
        self.nbSamplesUseMNIST = nbSamplesUseMNIST

        # Latent space size
        self.latent_space_size = latent_space_size

        # number of epochs
        self.epochs = epochs

        # batch size
        self.batch_size = batch_size

        # Learning rate
        self.lr = lr

        # Weight decay
        self.weight_decay = weight_decay

        # Loss function
        self.loss_function = loss_function
        if (loss_function.lower() == 'mse_loss'):
            self.criterion = torch.nn.MSELoss()
        elif (loss_function.lower() == 'bce_loss'):
            self.criterion = torch.nn.BCELoss()
        else:
            raise NotImplementedError("Loss function {} not yet supported".format(loss_function))

        # Loading the data
        self.dataset_name = dataset_name
        if (self.dataset_name.lower() == 'mnist'):
            transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(20), torchvision.transforms.ToTensor()])
            # Training data
            train_dataset = torchvision.datasets.MNIST(
                root="../datasets/", train=True, transform=transform, download=True
            )
            # Testing data
            self.train_dataset = [train_dataset[i] for i in range(len(train_dataset)) if i < self.nbSamplesUseMNIST ]
            print('Number of total samples for training: {}'.format(len(self.train_dataset)))
            #print("Size of one image: ", (self.train_dataset[0][0]).shape)
            self.test_dataset = torchvision.datasets.MNIST(
                root="../datasets/", train=False, transform=transform, download=True
            )
        elif (self.dataset_name.lower() == 'organcmnist'):
            self.train_data, self.val_data, self.test_data = loadFromHDF5('../datasets/organcmnist_0/data.hdf5') # HITS Dataset
            self.train_data = mix_data_splits(self.train_data, self.val_data)
            self.train_dataset = MedMNIST(self.train_data)
            self.test_dataset = MedMNIST(self.test_data)
        else:
            raise NotImplementedError('Dataset {} is not implemented'.format(self.dataset_name))

        # Data loaders
        self.train_loader = None
        self.test_loader = None
        print("=======> Finished Loading the data <=======\n")


        # Creating a model instance
        print("\n=======> Creating instace of the model <=======")
        # Device to use for computation
        if (device.lower() == 'cpu'):
            self.device = torch.device('cpu')
        elif (device.lower() == 'gpu'):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            print("WARNING: Device {} not supported, using CPU instead".format(device))
            self.device = torch.device('cpu')

        # Model
        if (self.dataset_name.lower() == 'mnist'):
            self.model = MnistConvolutionalAE(input_shape=(20,20), latent_space_size=self.latent_space_size)
            nb_channels, h_in, w_in = 1, 20, 20
        elif (self.dataset_name.lower() == 'organcmnist'):
            nb_channels, h_in, w_in = 1, 28, 28
            self.model = MnistConvolutionalAE(input_shape=(28,28), latent_space_size=self.latent_space_size)
        self.model.apply(weights_init)
        summary(self.model.to(self.device), (nb_channels, w_in, h_in))
        self.model.double()
        self.model.to(self.device)

        # Optimizer
        self.optimizer = None

        # Compressed representations
        self.train_representations = [{'compressed_representation': [], 'original_image': [], 'label': []} for _ in range(self.epochs)]
        self.test_representations = [{'compressed_representation': [], 'original_image': [], 'label': []} for _ in range(self.epochs)]
        print("=======> Finished Creating instace of the model <=======\n")

    def initialize_optimizer(self):
        # Optimizer
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def initialize_data_loaders(self):
        # Creating the data loaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    def getSample(self, set_type):
        """
            Gives a randomly chosen sample from the selected set (Train or Test)
        """
        if (set_type.lower() == 'train'):
            sampleID = randint(0, len(self.train_dataset))
            sample = self.train_dataset[sampleID]
        elif (set_type.lower() == 'test'):
            sampleID = randint(0, len(self.test_dataset))
            sample = self.test_dataset[sampleID]
        else:
            raise ValueError("Set type {} not valid".format(set_type))

        # Getting the data and label of the sample
        sampleData, sampleLabel = sample[0], sample[1]

        return sampleData, sampleLabel

    def train_holdout(self):
        """
            Function to train and evaluate the model using the Holdout method
        """
        train_loss_vals = []
        test_loss_vals = []
        for epoch in range(self.epochs):
            self.model.train()
            print("Epoch {}".format(epoch))
            tmp_losses = []
            for sample in self.train_loader:
                data, labels = sample[0], sample[1]
                data = data.type(torch.DoubleTensor).to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, data)

                loss_data = loss.data.cpu()
                tmp_losses.append(loss_data.numpy())

                # Saving the compressed representation
                self.train_representations[epoch]['compressed_representation'].append(self.model.compressed_representation.cpu().detach())
                self.train_representations[epoch]['label'].append(labels)
                self.train_representations[epoch]['original_image'].append(data.cpu().detach())


                # Backward pass
                loss.backward()

                # Optimize: Updating weights
                self.optimizer.step()

            # Training metrics
            mean_train_loss = np.array(tmp_losses).mean()
            train_loss_vals.append(mean_train_loss)

            # Test
            test_loss, self.test_representations[epoch] = computeTestLossAE(self.model, self.test_loader, self.criterion, self.device)
            test_loss_vals.append(test_loss)

        # Final train and test losses
        self.train_loss_vals, self.test_loss_vals = train_loss_vals, test_loss_vals


    def save_state(self):
        # Saving the model
        # Creating the folder that will contain the model results
        inc = 0
        while os.path.isdir('../models/{}_{}_{}'.format(self.dataset_name, self.exp_id, inc)):
            inc +=1
        os.mkdir('../models/{}_{}_{}'.format(self.dataset_name, self.exp_id, inc))
        os.mkdir('../models/{}_{}_{}/Model'.format(self.dataset_name, self.exp_id, inc))
        os.mkdir('../models/{}_{}_{}/CompressedRepresentations'.format(self.dataset_name, self.exp_id, inc))

        # Saving the trained model
        modelPath = '../models/{}_{}_{}/Model/model.pth'.format(self.dataset_name, self.exp_id, inc)
        torch.save(self.model, modelPath)
        print("Model saved at: ", modelPath)

        # Saving the compressed representation
        # Training points
        trainPointsFileName = "../models/{}_{}_{}/CompressedRepresentations/training_representations.pth".format(self.dataset_name, self.exp_id, inc)
        print('Train Compressed Representation Path: {}'.format(trainPointsFileName))
        with open(trainPointsFileName, "wb") as fp:   #Pickling
            pickle.dump(self.train_representations, fp)
        # Testing points
        testPointsFileName = "../models/{}_{}_{}/CompressedRepresentations/testing_representations.pth".format(self.dataset_name, self.exp_id, inc)
        print('Test Compressed Representation Path: {}'.format(testPointsFileName))
        with open(testPointsFileName, "wb") as fp:   #Pickling
            pickle.dump(self.test_representations, fp)

        # Saving the training and testing metrics
        train_dict = {
                        'loss': self.train_loss_vals
                     }
        test_dict = {
                        'loss': self.test_loss_vals
                    }
        metricsPath = '../models/{}_{}_{}/Model/metrics.hdf5'.format(self.dataset_name, self.exp_id, inc)
        metricsPath = save_metrics(train_dict, test_dict, nameFile=metricsPath)
        print("Metrics saved at: ", metricsPath)

    def plotLosses(self):
        # Putting the losses under the right format to be able to use the
        # function plotMetrics from utils.plot_metrics_AE
        train_dict = {
                        'loss': self.train_loss_vals
                     }
        test_dict = {
                        'loss': self.test_loss_vals
                    }

        # Plotting the losses
        plotMetrics(train_dict, test_dict, 'Loss')


    def plotReconstruction(self, input_data, input_label, title):
        """
            Plot the reconstruction of the input when passe to the current model

            Arguments:
            ----------
            input_data: tensor or array
                Input (image) data to give to the model.
            input_label: int
                Label of the input data
        """
        # Creating subplots
        fig, axs = plt.subplots(1, 2)

        # Getting the original sample
        # As the data got from the dataset is under the form (C, W, H) and C = 1
        # we have to change it to the form (W, H) to be able to plot it using
        # matplotlib.pyplot.imshow()
        original_sample = input_data.reshape(input_data.shape[1], input_data.shape[2])
        axs[0].imshow(original_sample)
        axs[0].set_title("Original input")
        axs[0].axis("off")

        # Getting the reconstruction
        if (self.dataset_name.lower() == 'mnist'):
            reconstruction_sample = self.model(input_data.view(1, 1, 20, 20).type(torch.DoubleTensor).to(self.device)).cpu().detach().numpy()
        if (self.dataset_name.lower() == 'organcmnist'):
            reconstruction_sample = self.model(input_data.view(1, 1, 28, 28).type(torch.DoubleTensor).to(self.device)).cpu().detach().numpy()
        reconstruction_sample = reconstruction_sample.reshape(input_data.shape[1], input_data.shape[2])
        axs[1].imshow(reconstruction_sample)
        axs[1].set_title("Reconstructed input")
        axs[1].axis("off")

        # Showing the plot
        fig.suptitle(title)
        plt.show()

#==============================================================================#
#==============================================================================#


def main():
    # =========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("--parameters_file", required=False, default="../parameters_files/default_parameters_AE.json", help="File containing the parameters of the experiment", type=str)
    args = vars(ap.parse_args())

    # Parameter file
    parameters_file = args['parameters_file']

    # =========================================================================#
    # Reading json file
    with open(parameters_file, 'r') as f:
        parameters_dict = json.load(f)

    # Name of the experiment
    if ('name' not in parameters_dict):
        parameters_dict['name'] = "Experiment_MNIST_feature_extraction"

    # Handling the parameters for training
    if ('nb_epochs' not in parameters_dict):
        parameters_dict['nb_epochs'] = 20

    if ('batch_size' not in parameters_dict):
        parameters_dict['batch_size'] = 32

    if ('lr' not in parameters_dict):
        parameters_dict['lr'] = 8e-2

    if ('weight_decay' not in parameters_dict):
        parameters_dict['weight_decay'] = 1e-5

    if ('loss_function' not in parameters_dict):
        parameters_dict['loss_function'] = 'MSE_loss'

    if ('latent_space_size' not in parameters_dict):
        parameters_dict['latent_space_size'] = 32

    if ('nbSamplesUseMNIST' not in parameters_dict):
        parameters_dict['nbSamplesUseMNIST'] = 15000

    if ('dataset_name' not in parameters_dict):
        parameters_dict['dataset_name'] = 'MNIST'

    # =========================================================================#
    # Running the experiment
    # Creating instance of the experiment
    my_expe = my_expe_class(
                            exp_id=parameters_dict['name'],\
                            epochs=parameters_dict['nb_epochs'],\
                            batch_size=parameters_dict['batch_size'],\
                            lr=parameters_dict['lr'],\
                            weight_decay=parameters_dict['weight_decay'],\
                            loss_function=parameters_dict['loss_function'],\
                            device='gpu',\
                            latent_space_size=parameters_dict['latent_space_size'],\
                            nbSamplesUseMNIST=parameters_dict['nbSamplesUseMNIST'],\
                            dataset_name=parameters_dict['dataset_name']
                           )

    # Initializing the data loaders and optimizer
    my_expe.initialize_data_loaders()
    my_expe.initialize_optimizer()

    # Training the model
    print("\n============Training and Evaluating the model (HOLDOUT)============\n")
    my_expe.train_holdout()

    print("\n============Saving the results and the model============\n")
    # Saving the results and the model
    my_expe.save_state()

    print("\n============Plotting the training and testing lossses============\n")
    # Plotting the train and test losses
    my_expe.plotLosses()


    print("\n============Plotting some reconstructions using the trained AE============\n")
    # Visualizing the representation created by the model
    my_expe.model.eval()

    # Training sample example
    training_sample_data, training_sample_label = my_expe.getSample(set_type='Train')
    my_expe.plotReconstruction(training_sample_data, training_sample_label, "Training sample")

    # Testing sample example
    testing_sample_data, testing_sample_label = my_expe.getSample(set_type='Test')
    my_expe.plotReconstruction(testing_sample_data, testing_sample_label, "Testing sample")


#==============================================================================#
#==============================================================================#

if __name__=='__main__':
    main()
