#!/usr/bin/env python3
"""
    This code evaluates a label propagation method through a classification
    task. It also allows to use robust loss functions to partially compensate
    the label-noise introduced by automatic label propagation.

    Options:
    --------
        *--parameters_file: Path to a parameters file containing the different
        parameters of the experiment. These parameters are:
            -expID: Name of the experiment.
            -propagation_method: Method used to propagate the labels. Four
            methods are available: LQ-KNN, Std-KNN, OPF-Semi and NoProp.
            -percentageLabelsKeep: Percentage of samples of the training set
            that are going to be considered as originally labeled. The rest
            of the training samples are going to be considered as unlabeled
            and they are going to be labeled using a labele propagation method.
            -K: Neighborhood to consider for LQ-KNN and Std-KNN label propagation
            -ks: Error significance for LQ-KNN
            -kt: Error tolerance for LQ-KNN
            -localQualThresh: Threshold used by LQ-kNN to select the "good" local quality samples
            -sortedQualities: If True, the samples are going to be sorted by decreasing
            order of local quality. It is only used when propagating the labels with LQ-KNN
            -lr: learning rate.
            -weightDecay.
            -nbEpochs: Number of epochs.
            -batchSizeTrain.
            -batchSizeTest.
            -lossFunction: Loss function to use for training. Two options: 'CE'
            and 'GCE'
            -device: 'CPU' or 'GPU'.
            -nbRepetitions: Number of times to repeat the experiment for
            statistical purposes.

        *--projections_folder: Folder containing different sub-folders
        corresponding to the projections that we want to study. It is used to
        select the optimal projection to use for label propagation.

    The code generates a file with the metrics of the experiment (no model is
    saved as several models are trained based on the number of repetitions
    chosen). This file is stored in projections_folder/ClassificationResults/ (if
    the ClassificationResults folder does not exists, it is created by the code).
"""
import os
import json
import pickle
import argparse
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import matplotlib.pyplot as plt
import random
import numpy as np
from src.classification_model import MnistClassificationModel, GeneralizedCrossEntropy
from src.label_propagation import propagateLabels_LQKNN, propagateLabelsLocalQuality_LQKNN_withoutSort
from src.label_propagation import propagateLabels_StdKNN, propagateLabels_OPF
from src.optimal_projection_selection import select_optimal_projection
from utils.download_exps_data import download_label_propagation_with_classification_data

#==============================================================================#
#==============================================================================#
# MNIST Dataset Class
class MNISTLabelProp(Dataset):
    """
        MNIST Dataset Wrapper (usef to create the dataset from samples annotated
        with one chosen label propagation method)
    """
    def __init__(self, data):
        """
        Arguments:
        ----------
            data:
                The data structured in a dict where the key is the id of the sample
                and the value is another dict with two keys 'Data' and 'Label'
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        sample = self.data[idx]

        return sample['Data'], sample['Label']

#==============================================================================#
#==============================================================================#
# Classification experiment class
class ClassificationExperiment(object):
    def __init__(self,\
                 expID,\
                 projections_folder,\
                 propagation_method,\
                 percentageLabelsKeep,\
                 K,\
                 ks,\
                 kt,\
                 localQualThresh,\
                 sortedQualities,\
                 lr,\
                 weightDecay,\
                 nbEpochs,\
                 batchSizeTrain,\
                 batchSizeTest,\
                 lossFunction,\
                 device):
        """
            Class that compares two semi-automatic annotatio methods. The first
            one uses KNN and the local quality of samples to propagate labels
            and the second one uses only KNN

            Arguments:
            ----------
            expID: str
                Name of the experiment.
            projections_folder: str
                Folder containing different sub-folders
                corresponding to the projections that we want to study. It is used to
                select the optimal projection to use for label propagation.
            propagation_method: str
                Method used to propagate the labels. Three methods are available:
                LQ-KNN, Std-KNN and OPF-Semi.
            percentageLabelsKeep: float
                Percentage of samples of the training set that are going to be
                considered as originally labeled. The rest of the training
                samples are going to be considered as unlabeled and they are
                going to be labeled using a labele propagation method.
            K: int
                Neighborhood to consider for LQ-KNN and Std-KNN label propagation
            ks: float
                Error significance for LQ-KNN
            kt: float
                Error tolerance for LQ-KNN
            localQualThresh: float
                Threshold used by LQ-kNN to select the "good" local quality samples
            sortedQualities: bool
                If True, the samples are going to be sorted by decreasing
                order of local quality. It is only used when propagating
                the labels with LQ-KNN
            lr: float
                Learning rate.
            weightDecay: float
            nbEpochs: int
                Number of epochs.
            batchSizeTrain: int
            batchSizeTest: int
            lossFunction: str
                Loss function to use for training. Two options: 'CE' and 'GCE'
            device: str
                Device to do the computations 'CPU' or 'GPU'.
        """
        # Defining the name of the experiment
        self.expID = expID
        self.projectionsFolder = projections_folder
        self.propagationMethod = propagation_method
        self.percentageLabelsKeep = 0.1
        self.K = K
        self.ks, self.kt = ks, kt
        self.localQualThresh = localQualThresh
        self.sortedQualities = sortedQualities
        self.lr = lr
        self.weightDecay = weightDecay
        self.nbEpochs = nbEpochs
        self.batchSizeTrain = batchSizeTrain
        self.batchSizeTest = batchSizeTest
        self.nbClasses = 10
        # Loss function
        if (lossFunction.lower() == 'ce'):
            # Using Cross Entropy Loss
            self.criterion = torch.nn.CrossEntropyLoss()
        elif (lossFunction.lower() == 'gce'):
            # Using Generalized Cross Entropy Loss
            self.criterion = GeneralizedCrossEntropy()
        # Device
        if (device.lower() == 'cpu'):
            self.device = torch.device("cpu")
        elif (device.lower() == 'gpu'):
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            raise ValueError("Value {} for device is not valid".format(device))


    def getSamplesToAnnotate(self):
        """
            Creates a list of the status of the samples, dividing them between
            labeled an unlabeled samples. This list is obtained when computing
            the optimal projection. We have to do the selection of the labeled
            samples BEFORE the selection of the optimal projection to avoid
            having a biased optimal projection because of the use of ALL the
            available samples.

            Returns:
            --------
                labeled_samples: list
                    List where each element is a dictionary with three keys:
                    'Data', 'Label' and 'LocalQuality'
                unlabeled_samples: list
                    Same as labeled_samples. However, if the label of the
                    unlabeled sample is not known, we define it as None
        """
        # Getting the optimal projection, the labeled samples indices
        # and the unlabeled samples indices
        optimal_projection_results = select_optimal_projection(self.projectionsFolder, self.percentageLabelsKeep)
        best_projection_folder = optimal_projection_results['BestProjection']
        labeled_samples_idxs = optimal_projection_results['LabeledSamplesIdx']
        unlabeled_samples_idxs = optimal_projection_results['UnlabeledSamplesIdx']

        # Loading the data, labels and local qualities
        images_file = best_projection_folder + '/images_0.pth'
        data_file = best_projection_folder + '/representations_0.pth'
        labels_file = best_projection_folder + '/labels_0.pth'
        local_quality_file = best_projection_folder + '/localQuality_ks{}_kt{}_0.pth'.format(self.ks, self.kt)
        with open(images_file, "rb") as fp:
            images = pickle.load(fp)
        with open(data_file, "rb") as fp:
            data_points = pickle.load(fp)
        with open(labels_file, "rb") as fp:   # Unpickling
            labels = pickle.load(fp)
        # Local qualities
        if (not os.path.isfile(local_quality_file)):
            if (self.propagationMethod.lower() == 'lq-knn'):
                print("\nWARNING !!! No local quality file found for ks = {} and kt = {}; We are going to compute it !\n".format(self.ks, self.kt))
                print("========> Starting computation of the local quality <========")
                latent_space_repr = '/'.join(best_projection_folder.split('/')[:-3])
                latent_space_repr = latent_space_repr + '/CompressedRepresentations/training_representations.pth'
                with subprocess.Popen(\
                                        [
                                            'python',\
                                            '../src/projection_metrics.py',\
                                            '--projections_folder',\
                                            best_projection_folder,\
                                            '--latent_space_repr',\
                                            latent_space_repr,
                                            '--ks',\
                                            str(self.ks),
                                            '--kt',\
                                            str(self.kt),
                                            '--quality_lueks',\
                                            "True",
                                            '--local_quality',\
                                            "True"
                                        ], stdout=subprocess.PIPE
                                     ) as proc:
                    # Seing if the sample was annotated
                    for line in proc.stdout:
                        line = line.decode("utf-8")
                        # print(line)
                print("========> Finishing computation of the local quality <========")
        with open(local_quality_file, "rb") as fp:   # Unpickling
            local_qualities = pickle.load(fp)
            # NORMALIZING THE LOCAL QUALITIES BY THE MAX!!!!!!
            tmpMaxLocalQuals = max(local_qualities)
            local_qualities = [localQual/tmpMaxLocalQuals for localQual in local_qualities]


        # Separating the samples between labeled and unlabeled
        labeled_samples, unlabeled_samples = [], []
        for labeled_idx in labeled_samples_idxs:
            labeled_sample = {
                                'Data': data_points[labeled_idx],
                                'Label': labels[labeled_idx],
                                'LocalQuality': local_qualities[labeled_idx],
                                'Image': images[labeled_idx]
                             }
            labeled_samples.append(labeled_sample)
        for unlabeled_idx in unlabeled_samples_idxs:
            unlabeled_sample = {
                                'Data': data_points[unlabeled_idx],
                                'Label': labels[unlabeled_idx],
                                'LocalQuality': local_qualities[unlabeled_idx],
                                'Image': images[unlabeled_idx]
                             }
            unlabeled_samples.append(unlabeled_sample)

        return labeled_samples, unlabeled_samples




    def propagateLabels(self):
        """
            Proagate labels from the labeled samples to the unlabeled samples
            using a selected label propagation method.

            Does label propagation using of of the chosen methods. Three methods
            are available: LQ-KNN, Std-KNN and OPF-Semi.
        """
        # Getting the labeled and unlabeled samples
        labeled_samples, unlabeled_samples = self.getSamplesToAnnotate()
        # print("Percentage of labeled samples: ", 100*len(labeled_samples)/(len(labeled_samples) + len(unlabeled_samples)))
        # print("Percentage of unlabeled samples: ", 100*len(unlabeled_samples)/(len(labeled_samples) + len(unlabeled_samples)))

        # Doing the label propagation
        if (self.propagationMethod.lower() == 'opf-semi'):
            new_annotated_samples,\
            accuracy_annotation,\
            nb_annotated_samples,\
            total_number_of_samples,\
            number_initial_labeled_samples = propagateLabels_OPF(labeled_samples, unlabeled_samples)
        elif (self.propagationMethod.lower() == 'lq-knn'):
            try:
                if (self.sortedQualities):
                    prop_method = propagateLabels_LQKNN
                else:
                    prop_method = propagateLabelsLocalQuality_LQKNN_withoutSort
            except:
                raise KerError("self.sortedQualities attribute was not found, please indicate it when doing LQ-KNN propagation")
            new_annotated_samples,\
            accuracy_annotation,\
            nb_annotated_samples,\
            total_number_of_samples,\
            number_initial_labeled_samples = prop_method(labeled_samples, unlabeled_samples, self.K, self.localQualThresh)
        elif (self.propagationMethod.lower() == 'std-knn'):
            new_annotated_samples,\
            accuracy_annotation,\
            nb_annotated_samples,\
            total_number_of_samples,\
            number_initial_labeled_samples = propagateLabels_StdKNN(labeled_samples, unlabeled_samples, self.K)
        elif (self.propagationMethod.lower() == 'noprop'):
            new_annotated_samples = labeled_samples
            accuracy_annotation = None
            nb_annotated_samples = 0
            total_number_of_samples = len(labeled_samples)
            number_initial_labeled_samples = len(labeled_samples)
        else:
            raise ValueError("Propagation mode {} is not supported".format(self.propagationMethod))


        return labeled_samples,\
               new_annotated_samples,\
               accuracy_annotation,\
               nb_annotated_samples,\
               total_number_of_samples,\
               number_initial_labeled_samples

    def createTrainLoader(self, labeled_samples, new_labeled_samples):
        """
            Creates the train data laoder for classification.

            Arguments:
            ----------
                labeled_samples: list
                    List of samples. Each sample is a dict with three keys:
                    'Data', 'Label', 'LocalQuality'
                new_labeled_samples: list
                    Same as labeled_samples but this samples are obtained with
                    a label propagation method
        """
        # Creating the data dictionary
        data = {}
        currentID = 0
        for sample in labeled_samples:
            data[currentID] = {'Data':sample['Image'], 'Label':sample['Label']}
            currentID += 1
        for sample in new_labeled_samples:
            data[currentID] = {'Data':sample['Image'], 'Label':sample['Label']}
            currentID += 1

        # Creating the train dataset
        trainDataset = MNISTLabelProp(data)
        trainIndices = list(range(0, len(trainDataset)))
        trainSampler = SubsetRandomSampler(trainIndices)
        self.trainLoader = torch.utils.data.DataLoader(trainDataset,\
                                                  batch_size=self.batchSizeTrain,\
                                                  sampler=trainSampler)

    def createTestLoader(self):
        """
            Creates the test data loader for classification.
        """
        transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(20), torchvision.transforms.ToTensor()])
        testDataset = torchvision.datasets.MNIST(
            root="~/torch_datasets", train=False, transform=transform, download=True
        )
        self.testLoader = torch.utils.data.DataLoader(
            testDataset, batch_size=self.batchSizeTest, shuffle=False, num_workers=4
        )

    def trainModel(self):
        """
            Trains a classification model (defined in src.classification_model)
            using a dataset created with a label propagation method. The
            test set is NOT obtained with a label propagation method, it is
            the test set defined by torchvision.
        """
        # Model
        model = MnistClassificationModel()
        model.double()
        model = model.to(self.device)

        # Optimizer
        optimizer = torch.optim.Adamax(model.parameters(), lr=self.lr, weight_decay=self.weightDecay)

        # Data structure for the losses and test accuracy
        train_loss_vals = []
        train_accuracy_vals = []
        test_loss_vals = []
        test_accuracy_vals = []

        # Doing the training
        for epoch in range(self.nbEpochs):
            model.train()
            print("Epoch {}".format(epoch))
            tmp_losses = []
            correct_train = 0
            for sample in self.trainLoader:
                # Getting the data and the labels and putting them in the right
                # device for the computations
                data, labels = sample[0], sample[1]
                data = data.type(torch.DoubleTensor).to(self.device)
                labels = labels.to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                output = model(data)
                loss = self.criterion(output, labels)
                loss_data = loss.data.cpu()
                tmp_losses.append(loss_data.numpy())
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct_train += pred.eq(labels.view_as(pred)).sum().item()

                # Backward pass
                loss.backward()

                # Optimize: Updating weights
                optimizer.step()

            # Training metrics
            mean_train_loss = np.array(tmp_losses).mean()
            train_loss_vals.append(mean_train_loss)
            trainAccuracy = 100. * correct_train / len(self.trainLoader.dataset)
            train_accuracy_vals.append(trainAccuracy)

            # Evaluation
            model.eval()
            tmp_test_loss = []
            correct_test = 0
            with torch.no_grad():
                for data, target in self.testLoader:
                    data, target = data.type(torch.DoubleTensor).to(self.device), target.to(self.device)
                    output = model(data)
                    test_loss = self.criterion(output, target)
                    test_loss_data = test_loss.data.cpu()
                    tmp_test_loss.append(test_loss_data.numpy())
                    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                    correct_test += pred.eq(target.view_as(pred)).sum().item()
            mean_test_loss = np.array(tmp_test_loss).mean()
            test_loss_vals.append(mean_test_loss)
            testAccuracy = 100. * correct_test / len(self.testLoader.dataset)
            test_accuracy_vals.append(testAccuracy)
            print("Test accuracy at epoch {}: {}".format(epoch, testAccuracy))

        return {
                    'TrainLoss': train_loss_vals,
                    'TrainAcc': train_accuracy_vals,
                    'TestLoss': test_loss_vals,
                    'TestAcc': test_accuracy_vals
                }

#==============================================================================#
#==============================================================================#
# Plot functions
def plotMetricsClassification(results, metric_name):
    """
        Plots for the classification results according to one metric.

        Arguments:
        ----------
        results: list
            List where element i corresponds to the results of repetition i
            of the classification experiment. results[i] is a dictionary with
            four keys: TrainLoss, TrainAcc, TestLoss, TestAa. Finally,
            results[i][metric_name][j] corresponds to the value of metric_name
            at epoch j for repetition i
        metric_name: str
            Name of the metric to plot
    """
    # Number of epochs
    nbEpochs = len(results[0]['Train'+metric_name])

    data_train_means, data_train_stds = [], []
    data_test_means, data_test_stds = [], []
    for epoch in range(nbEpochs):
        # Train statistics results
        data_train = [results[i]['Train'+metric_name][epoch] for i in range(len(results))]
        data_train_means.append(np.mean(data_train))
        data_train_stds.append(np.std(data_train))
        # Test statistics results
        data_test = [results[i]['Test'+metric_name][epoch] for i in range(len(results))]
        data_test_means.append(np.mean(data_test))
        data_test_stds.append(np.std(data_test))

    # Doing the boxplot
    plt.errorbar(list(range(nbEpochs)), data_train_means, yerr=data_train_stds, label='Train'+metric_name)
    plt.errorbar(list(range(nbEpochs)), data_test_means, yerr=data_test_stds, label='Test'+metric_name)
    plt.title(metric_name)
    plt.xlabel("Epochs")
    plt.ylabel(metric_name)
    plt.legend()

    plt.show()



#==============================================================================#
#==============================================================================#

def main():
    #==========================================================================#
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    default_projections_folder = '../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/'
    ap.add_argument("--parameters_file", required=False, default="../parameters_files/default_parameters_classification.json", help="File containing the parameters of the experiment", type=str)
    ap.add_argument("--projections_folder", default=default_projections_folder, help="Folder to the files describing the embedded data", type=str)
    args = vars(ap.parse_args())

    # Parameter file
    parameters_file = args['parameters_file']
    projections_folder = args['projections_folder']

    #==========================================================================#
    #==========================================================================#
    # Loading the parameters from the parameters file
    with open(parameters_file, 'r') as f:
        parameters_dict = json.load(f)

    # Name of the experiment
    if ('expID' not in parameters_dict):
        parameters_dict['expID'] = "Experiment_Emboli_Classification"

    if ('propagation_method' not in parameters_dict):
        parameters_dict['propagation_method'] = "LQ-KNN"

    if ('percentageLabelsKeep' not in parameters_dict):
        parameters_dict['percentageLabelsKeep'] = 0.1

    if ('K' not in parameters_dict):
        parameters_dict['K'] = 10

    if ('ks' not in parameters_dict):
        parameters_dict['ks'] = 10

    if ('kt' not in parameters_dict):
        parameters_dict['kt'] = 10

    if ('localQualThresh' not in parameters_dict):
        parameters_dict['localQualThresh'] = 0.1

    if ('sortedQualities' not in parameters_dict):
        parameters_dict['sortedQualities'] = True
    else:
        if (parameters_dict['sortedQualities'].lower() == 'true'):
            parameters_dict['sortedQualities'] = True
        elif (parameters_dict['sortedQualities'].lower() == 'false'):
            parameters_dict['sortedQualities'] = False
        else:
            raise ValueError("Value {} for the parameter sortedQualities is not valid".format(parameters_dict['sortedQualities']))

    if ('ks' not in parameters_dict):
        parameters_dict['ks'] = 10

    if ('lr' not in parameters_dict):
        parameters_dict['lr'] = 0.001

    if ('weightDecay' not in parameters_dict):
        parameters_dict['weightDecay'] = 1e-7

    if ('nbEpochs' not in parameters_dict):
        parameters_dict['nbEpochs'] = 50

    if ('batchSizeTrain' not in parameters_dict):
        parameters_dict['batchSizeTrain'] = 32

    if ('batchSizeTest' not in parameters_dict):
        parameters_dict['batchSizeTest'] = 64

    if ('lossFunction' not in parameters_dict):
        parameters_dict['lossFunction'] = "CE"

    if ('device' not in parameters_dict):
        parameters_dict['device'] = "GPU"

    if ('nbRepetitions' not in parameters_dict):
        parameters_dict['nbRepetitions'] = 10

    #==========================================================================#
    #==========================================================================#
    # If the default parameters are used, we area going to download the
    # useful data if it has not been done already
    if ('/MNIST_Example_0/' in projections_folder):
        download_label_propagation_with_classification_data()

    #==========================================================================#
    #==========================================================================#
    # Creating a classification experiment instance
    print("\n\n=========Creating instance of the classification experiment=========")
    my_expe = ClassificationExperiment(
                                         expID=parameters_dict['expID'],\
                                         projections_folder=projections_folder,\
                                         propagation_method=parameters_dict['propagation_method'],\
                                         percentageLabelsKeep=parameters_dict['percentageLabelsKeep'],\
                                         K=parameters_dict['K'],\
                                         ks=parameters_dict['ks'],\
                                         kt=parameters_dict['kt'],\
                                         localQualThresh=parameters_dict['localQualThresh'],\
                                         sortedQualities=parameters_dict['sortedQualities'],\
                                         lr=parameters_dict['lr'],\
                                         weightDecay=parameters_dict['weightDecay'],\
                                         nbEpochs=parameters_dict['nbEpochs'],\
                                         batchSizeTrain=parameters_dict['batchSizeTrain'],\
                                         batchSizeTest=parameters_dict['batchSizeTest'],\
                                         lossFunction=parameters_dict['lossFunction'],\
                                         device=parameters_dict['device'],\
                                    )
    print("=======> DONE")
    print("====================================================================")


    #==========================================================================#
    #==========================================================================#
    # Doing label propagation on the training set
    print("\n\n=========Creating the training set using label propagation=========")
    labeled_samples,\
    new_labeled_samples,\
    accuracy_annotation,\
    nb_annotated_samples,\
    total_number_of_samples,\
    number_initial_labeled_samples = my_expe.propagateLabels()
    print("=======>We annotated {} samples with an annotation accuracy of {}".format(nb_annotated_samples, accuracy_annotation))
    print("=======>In total we have {} training samples".format(nb_annotated_samples + number_initial_labeled_samples))
    print("=======> DONE")
    print("====================================================================")

    #==========================================================================#
    #==========================================================================#
    # Creating the data loaders
    print("\n\n==============Creating the train and test dataloaders==============")
    my_expe.createTrainLoader(labeled_samples, new_labeled_samples)
    my_expe.createTestLoader()
    print("=======> DONE")
    print("====================================================================")

    #==========================================================================#
    #==========================================================================#
    # Doing the training and testing
    print("\n\n================Training the model ({} repetitions)================".format(parameters_dict['nbRepetitions']))
    results = []
    for repetition in range(parameters_dict['nbRepetitions']):
        print("=======> Doing repetition {} <=======".format(repetition))
        tmp_res = my_expe.trainModel()
        results.append(tmp_res)
        print("Train accuracy {}, Test accuracy {}".format(tmp_res['TrainAcc'][-1], tmp_res['TestAcc'][-1]))
        print("\n")

    # Saving the results
    if (not os.path.isdir(projections_folder+'/ClassificationResults/')):
        os.mkdir(projections_folder+'/ClassificationResults/')
    inc = 0
    fileName = projections_folder + '/ClassificationResults/' + parameters_dict['expID'] + '_'
    while (os.path.isfile(fileName + str(inc) + '.pth')):
        inc += 1
    fileName = fileName + str(inc) + '.pth'
    with open(fileName, "wb") as fp:   #Pickling
        pickle.dump(results, fp)
    print("Results of the experiment saved at: {}".format(fileName))

    # Plotting the results of the classification
    plotMetricsClassification(results, 'Loss')
    plotMetricsClassification(results, 'Acc')

    print("=======> DONE")
    print("====================================================================")







if __name__=="__main__":
    main()
