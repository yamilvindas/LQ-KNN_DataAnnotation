#!/usr/bin/env python3
"""
    Evaluation of different label propagation methods on the MNIST dataset.

    Options:
    --------
        *--exp_ID: Name of the experiment
        *--projections_folder: Folder containing different sub-folders
        corresponding to the projections that we want to study. It is used to
        select the optimal projection to use for label propagation.
        *--propagation_mode: Mode to propagate labels (propLocalQual,
        classicalProp or OPF-Semi)
        *--var_to_study: Variable to study (K, percentageLabelsKeep or
        localQualThresh)
        *--sorted_qualities: True if wanted to sort the samples by local quality
         when propagating the labels using LQ-KNN
        *--local_quality_threshold: Local quality threshold to use if
        propLocalQual mode is used and the variable to study is not the local quality threshold
        *--ks: Value of ks to choose the local quality file to use for LQ-kNN
        *--kt: Value of kt to choose the local quality file to use for LQ-kNN
        *--projection_type_to_use: projection_type_to_use: Type of projection to use for
        label propagation. Three choices are possible: Best, Middle and Worst.
        If Best is chosen, then the best projection selected according
        to the Silhouette Score is used.
        If Worst is chose, then the worst projection selected according
        to the Silhouette Score is used.

    It stores the results in a folder named 'LabelPropResults' in the same folder
    as the projections folder. The name of the file is of the form
    expID_propMode-{}_var-to-study-{}.pth
"""

import os
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from time import time
import copy
from src.label_propagation import propagateLabels_LQKNN, propagateLabelsLocalQuality_LQKNN_withoutSort
from src.label_propagation import propagateLabels_StdKNN, propagateLabels_OPF
from src.optimal_projection_selection import select_optimal_projection
from utils.download_exps_data import download_label_propagation_data

class Experiment(object):
    def __init__(self,\
                 exp_ID,\
                 projections_folder,\
                 ks,\
                 kt,\
                 projection_type_to_use):
        """
            Class that compares two semi-automatic annotatio methods. The first
            one uses KNN and the local quality of samples to propagate labels
            and the second one uses only KNN

            Arguments:
            ----------
            projections_folder: str
                Folder containing different sub-folders corresponding to the
                projections that we want to study. It is used to select
                the optimal projection to use for label propagation.
            ks: int
                Value of ks to choose the local quality file to use for LQ-kNN
            kt: int
                Value of kt to choose the local quality file to use for LQ-kNN
            projection_type_to_use: str
                Type of projection to use for label propagation. Three choices
                are possible: Best, Middle and Worst.
                If Best is chosen, then the best projection selected according
                to the Silhouette Score is used.
                If Worst is chose, then the worst projection selected according
                to the Silhouette Score is used.
        """
        # Defining the name of the experiment
        self.expID = exp_ID

        # Defining the folder which contains the final embedded representations
        # of the original samples (in general they are 2D points) as well
        # as the local qualities for each embedded representation
        self.projectionsFolder = projections_folder

        # Defining the values of ks and kt to use for the local quality
        self.ks, self.kt = ks, kt

        # Projection to use
        self.projection_type_to_use = projection_type_to_use

        # Defining some attributes
        self.nbClasses = 10

    def getSamplesToAnnotate(self, percentageLabelsKeep):
        """
            Creates a list of the status of the samples, dividing them between
            labeled an unlabeled samples. This list is obtained when computing
            the optimal projection. We have to do the selection of the labeled
            samples BEFORE the selection of the optimal projection to avoid
            having a biased optimal projection because of the use of ALL the
            available samples.

            Arguments:
            ----------
            percentageLabelsKeep: float
                Float between 0 and 1 corresponding to the number of labeled
                samples to keep among all the labeled samples

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
        optimal_projection_results = select_optimal_projection(self.projectionsFolder, percentageLabelsKeep)
        if (self.projection_type_to_use.lower() == 'best'):
            projection_folder_to_use = optimal_projection_results['BestProjection']
        elif (self.projection_type_to_use.lower() == 'middle'):
            projection_folder_to_use = optimal_projection_results['MiddleProjection']
        elif (self.projection_type_to_use.lower() == 'worst'):
            projection_folder_to_use = optimal_projection_results['WorstProjection']
        labeled_samples_idxs = optimal_projection_results['LabeledSamplesIdx']
        unlabeled_samples_idxs = optimal_projection_results['UnlabeledSamplesIdx']

        # Loading the data, labels and local qualities
        data_file = projection_folder_to_use + '/representations_0.pth'
        labels_file = projection_folder_to_use + '/labels_0.pth'
        local_quality_file = projection_folder_to_use + '/localQuality_ks{}_kt{}_0.pth'.format(self.ks, self.kt)
        with open(data_file, "rb") as fp:
            data_points = pickle.load(fp)
        with open(labels_file, "rb") as fp:   # Unpickling
            labels = pickle.load(fp)
        # Local qualities
        if (not os.path.isfile(local_quality_file)):
            print("\nWARNING !!! No local quality file found for ks = {} and kt = {}; We are going to compute it !\n".format(self.ks, self.kt))
            print("========> Starting computation of the local quality <========")
            latent_space_repr = '/'.join(projection_folder_to_use.split('/')[:-3])
            latent_space_repr = latent_space_repr + '/CompressedRepresentations/training_representations.pth'
            with subprocess.Popen(\
                                    [
                                        'python',\
                                        '../src/projection_metrics.py',\
                                        '--projections_folder',\
                                        projection_folder_to_use,\
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
                                'LocalQuality': local_qualities[labeled_idx]
                             }
            labeled_samples.append(labeled_sample)
        for unlabeled_idx in unlabeled_samples_idxs:
            unlabeled_sample = {
                                'Data': data_points[unlabeled_idx],
                                'Label': labels[unlabeled_idx],
                                'LocalQuality': local_qualities[unlabeled_idx]
                             }
            unlabeled_samples.append(unlabeled_sample)

        return labeled_samples, unlabeled_samples


    def propagateLabels(self, propagation_mode,\
                        percentageLabelsKeep,
                        **kwargs):
        """
            Does label propagation using of of the chosen methods. Three methods
            are available: LQ-KNN, Std-KNN and OPF-Semi.

            Arguments:
            ----------
            propagation_mode: str
                Mode to propagate the labels. Three possible modes:
                    - propLocalQual for LQ-kNN
                    - classicalProp for Std-kNN
                    - OPF-Semi for Semi-supervised OPF
            percentageLabelsKeep: float
                Percentage of samples that are going to be considered as labeled
                among all the available labeled samples. The other samples
                are going to be considered as unlabeled
            kwargs:
                Other valid arguments are:
                    K: int
                        Neighborhood to consider for label propagatio when using
                        LQ-KNN and Std-KNN
                    localQualThresh: float
                        Threshold used by LQ-kNN to select the "good" local
                        quality samples
                    sorted_qualities: bool
                        If True, the samples are going to be sorted by decreasing
                        order of local quality. It is only used when propagating
                        the labels with LQ-KNN
        """
        # Verifying that the percentage of labeled samples to keep is betwee
        # 0 and 1
        assert (percentageLabelsKeep >= 0) and (percentageLabelsKeep <= 1)

        # Getting the labeled and unlabeled samples
        labeled_samples, unlabeled_samples = self.getSamplesToAnnotate(percentageLabelsKeep)
        # print("Percentage of labeled samples: ", 100*len(labeled_samples)/(len(labeled_samples) + len(unlabeled_samples)))
        # print("Percentage of unlabeled samples: ", 100*len(unlabeled_samples)/(len(labeled_samples) + len(unlabeled_samples)))

        # Doing the label propagation
        if (propagation_mode.lower() == 'opf-semi'):
            new_annotated_samples,\
            accuracy_annotation,\
            nb_annotated_samples,\
            total_number_of_samples,\
            number_initial_labeled_samples = propagateLabels_OPF(labeled_samples, unlabeled_samples)
        elif (propagation_mode.lower() == 'proplocalqual'):
            try:
                if (kwargs['sorted_qualities']):
                    prop_method = propagateLabels_LQKNN
                else:
                    prop_method = propagateLabelsLocalQuality_LQKNN_withoutSort
            except:
                raise KerError("sorted_qualities argument was not found, please indicate it when doing LQ-KNN propagation")
            new_annotated_samples,\
            accuracy_annotation,\
            nb_annotated_samples,\
            total_number_of_samples,\
            number_initial_labeled_samples = prop_method(labeled_samples, unlabeled_samples, kwargs['K'], kwargs['localQualThresh'])
        elif (propagation_mode.lower() == 'classicalprop'):
            new_annotated_samples,\
            accuracy_annotation,\
            nb_annotated_samples,\
            total_number_of_samples,\
            number_initial_labeled_samples = propagateLabels_StdKNN(labeled_samples, unlabeled_samples, kwargs['K'])
        else:
            raise ValueError("Propagation mode {} is not supported".format(propagation_mode))


        return new_annotated_samples,\
               accuracy_annotation,\
               nb_annotated_samples,\
               total_number_of_samples,\
               number_initial_labeled_samples

def plotResults(results, var_to_study):
    """
        Plot the results for the label propagation experiment.
    """
    # Values of the x-axis
    x_vals = list(results['accs'].keys())
    x_vals.sort()

    # Values of the y-axis
    y_mean_annot_acc, y_std_annot_acc = [], []
    y_mean_nb_labeled_samples, y_std_nb_labeled_samples = [], []
    for x in x_vals:
        # Annot accs
        y_mean_annot_acc.append(np.mean(results['accs'][x]))
        y_std_annot_acc.append(np.std(results['accs'][x]))
        # Nb labeled samples
        y_mean_nb_labeled_samples.append(np.mean(results['nbsAnnotatedSamples'][x]))
        y_std_nb_labeled_samples.append(np.std(results['nbsAnnotatedSamples'][x]))

    # Matplotib fontsize parameter
    plt.rcParams.update({'font.size': 20})

    # Annotation accuracy
    plt.errorbar(x_vals, y_mean_annot_acc, yerr=y_std_annot_acc)
    plt.xlabel(var_to_study)
    plt.ylabel("Annotation accuracy")
    plt.legend()
    plt.show()
    # Number of labeled samples
    plt.errorbar(x_vals, y_mean_nb_labeled_samples, yerr=y_std_nb_labeled_samples)
    plt.xlabel(var_to_study)
    plt.ylabel("Nb Labeled Samples")
    plt.legend()
    plt.show()

#==============================================================================#
#==============================================================================#

def main():
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    default_projections_folder = '../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/'
    ap.add_argument('--exp_ID', default='evaluation_label_propagation_MNIST', help="Name of the experiment", type=str)
    ap.add_argument("--projections_folder", default=default_projections_folder, help="Folder to the files describing the embedded data", type=str)
    ap.add_argument('--propagation_mode', default='propLocalQual', help="Mode to propagate labels (propLocalQual or classicalProp or OPF-Semi)", type=str)
    ap.add_argument('--var_to_study', default='K', help="Variable to study (K, percentageLabelsKeep or localQualThresh)", type=str)
    ap.add_argument('--sorted_qualities', default='True', help="True if wanted to sort the samples by local quality when propagating the labels using LQ-KNN", type=str)
    ap.add_argument('--local_quality_threshold', default=0.1, help="Local quality threshold to use if propLocalQual mode is used and the variable to study is not the local quality threshold", type=float)
    ap.add_argument('--ks', default=10, help="Value of ks to choose the local quality file to use for LQ-kNN", type=str)
    ap.add_argument('--kt', default=10, help="Value of kt to choose the local quality file to use for LQ-kNN", type=str)
    ap.add_argument('--projection_type_to_use', default='Best', help="Type of projection to use for label propagation. Three choices are possible: Best, Middle and Worst", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    exp_ID = args['exp_ID']
    projections_folder = args['projections_folder']
    propagation_mode = args['propagation_mode']
    print("\n=======> Doing {} label propagation <=======\n".format(propagation_mode))
    var_to_study = args['var_to_study']
    localQualThresh = args['local_quality_threshold']
    sorted_qualities = args['sorted_qualities']
    if (sorted_qualities == 'True' or sorted_qualities == 'true'):
        sorted_qualities = True
    elif (sorted_qualities == 'False' or sorted_qualities == 'false'):
        sorted_qualities = False
    else:
        raise ValueError("Option --sorted_qualities cannot take the value {}".format(sorted_qualities))
    ks, kt = args['ks'], args['kt']
    projection_type_to_use = args['projection_type_to_use']
    print("Using the {} projection for label propagation".format(projection_type_to_use))


    #==========================================================================#
    # If the default parameters are used, we area going to download the
    # useful data if it has not been done already
    if ('/MNIST_Example_0/' in projections_folder):
        download_label_propagation_data()


    #==========================================================================#
    # Initializing the experiment
    exp = Experiment(exp_ID, projections_folder, ks, kt, projection_type_to_use)

    #==========================================================================#
    results = {}
    if (propagation_mode.lower() == 'opf-semi'):
        nbRepetitions = 30
    else:
        nbRepetitions = 100 # For statistical purposes
    if (var_to_study is None): # In this case we do OPF-Semi label propagation
        # Verifying that the propagation mode is OPF-Semi
        if (propagation_mode.lower() != 'opf-semi'):
            raise ValueError("If var_to_study is None, the only supported propagation mode is OPF-Semi")
        print("\n=======>No particular variable is being studied. Doing OPF-semi label prop")

        # Starting the experiment
        executionTimes = []
        percentageLabelsKeep = 0.1
        results = {
                    'accs': [],
                    'nbsAnnotatedSamples': [],
                    'percentageLabelsKeep': percentageLabelsKeep,
                    'initNbAnnotateSamples': [],
                    'nbTotalSamples': [],
                    }
        for repetition_number in range(nbRepetitions):
            print("=======> Repetition n°{} <=======".format(repetition_number))
            startTime = time()

            # Doing the propagation
            newAnnotatedSamples,\
            accuracyAnnotation,\
            nbAnnotatedSamples,\
            totNumberSamples,\
            initNbAnnotateSamples = exp.propagateLabels(propagation_mode,\
                                                        percentageLabelsKeep)
            # Saving the results
            results['accs'].append(accuracyAnnotation)
            results['nbsAnnotatedSamples'].append(nbAnnotatedSamples)
            results['initNbAnnotateSamples'].append(initNbAnnotateSamples)
            results['nbTotalSamples'].append(totNumberSamples)
            endTime = time()
            executionTimes.append(endTime-startTime)
        print("Needed time to do the propagation with {} for K= {}: {} +- {} s".format(propagationMode, K, np.mean(executionTimes), np.std(executionTimes)))

    elif (var_to_study == 'K'):
        print("\n=======>Variable studied: {}".format(var_to_study))
        K_vals = [i for i in range(21)]
        percentageLabelsKeep = 0.1
        results = {
                    'K': K_vals.copy(),
                    'accs': {},
                    'nbsAnnotatedSamples': {},
                    'localQualThresh': localQualThresh,
                    'percentageLabelsKeep': percentageLabelsKeep,
                    'initNbAnnotateSamples': {},
                    'nbTotalSamples': {}
                    }
        for K in K_vals:
            print("\t===> Value of K = {}".format(K))
            executionTimes = []
            for repetition_number in range(nbRepetitions):
                print("=======> Repetition n°{} <=======".format(repetition_number))
                startTime = time()
                # Doing the propagation
                newAnnotatedSamples,\
                accuracyAnnotation,\
                nbAnnotatedSamples,\
                totNumberSamples,\
                initNbAnnotateSamples = exp.propagateLabels(propagation_mode,\
                                                            percentageLabelsKeep,
                                                            K=K,\
                                                            localQualThresh=localQualThresh,\
                                                            sorted_qualities=sorted_qualities)

                # Saving the results
                if (K not in results['accs']):
                    results['accs'][K] = [accuracyAnnotation]
                    results['nbsAnnotatedSamples'][K] = [nbAnnotatedSamples]
                    results['initNbAnnotateSamples'][K] = [initNbAnnotateSamples]
                    results['nbTotalSamples'][K] = [totNumberSamples]
                else:
                    results['accs'][K].append(accuracyAnnotation)
                    results['nbsAnnotatedSamples'][K].append(nbAnnotatedSamples)
                    results['initNbAnnotateSamples'][K].append(initNbAnnotateSamples)
                    results['nbTotalSamples'][K].append(totNumberSamples)
                endTime = time()
                executionTimes.append(endTime-startTime)
            print("Needed time to do the propagation with {} for K= {}: {} +- {} s".format(propagation_mode, K, np.mean(executionTimes), np.std(executionTimes)))

    elif (var_to_study == 'percentageLabelsKeep'):
        print("\n=======>Variable studied: {}".format(var_to_study))
        percentageLabelsKeep_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        K = 5
        results = {
                    'K': K,
                    'accs': {},
                    'nbsAnnotatedSamples': {},
                    'localQualThresh': localQualThresh,
                    'percentageLabelsKeep': percentageLabelsKeep_vals.copy(),
                    'initNbAnnotateSamples': {},
                    'nbTotalSamples': {}
                    }
        for percentageLabelsKeep in percentageLabelsKeep_vals:
            print("\t===> Value of percentageLabelsKeep = {}".format(percentageLabelsKeep))
            for repetition_number in range(nbRepetitions):
                print("=======> Repetition n°{} <=======".format(repetition_number))
                # Doing the propagation
                newAnnotatedSamples,\
                accuracyAnnotation,\
                nbAnnotatedSamples,\
                totNumberSamples,\
                initNbAnnotateSamples = exp.propagateLabels(propagation_mode,\
                                                             percentageLabelsKeep,
                                                             K=K,\
                                                             localQualThresh=localQualThresh,\
                                                             sorted_qualities=sorted_qualities)

                # Saving the results
                if (percentageLabelsKeep not in results['accs']):
                    results['accs'][percentageLabelsKeep] = [accuracyAnnotation]
                    results['nbsAnnotatedSamples'][percentageLabelsKeep] = [nbAnnotatedSamples]
                    results['initNbAnnotateSamples'][percentageLabelsKeep] = [initNbAnnotateSamples]
                    results['nbTotalSamples'][percentageLabelsKeep] = [totNumberSamples]
                else:
                    results['accs'][percentageLabelsKeep].append(accuracyAnnotation)
                    results['nbsAnnotatedSamples'][percentageLabelsKeep].append(nbAnnotatedSamples)
                    results['initNbAnnotateSamples'][percentageLabelsKeep].append(initNbAnnotateSamples)
                    results['nbTotalSamples'][percentageLabelsKeep].append(totNumberSamples)

    elif (var_to_study == 'localQualThresh'):
        print("\n=======>Variable studied: {}".format(var_to_study))
        if (propagation_mode != 'propLocalQual'):
            raise ValueError("If var_to_study is 'localQualThresh', then mode has to be 'propLocalQual'" )
        else:
            localQualThresh_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            percentageLabelsKeep = 0.1
            K = 5
            results = {
                        'K': K,
                        'accs': {},
                        'nbsAnnotatedSamples': {},
                        'localQualThresh': localQualThresh_vals.copy(),
                        'percentageLabelsKeep': percentageLabelsKeep,
                        'initNbAnnotateSamples': {},
                        'nbTotalSamples': {}
                        }
            for localQualThresh in localQualThresh_vals:
                print("\t===> Value of localQualThresh = {}".format(localQualThresh))
                for repetition_number in range(nbRepetitions):
                    print("=======> Repetition n°{} <=======".format(repetition_number))
                    # Doing the propagation
                    newAnnotatedSamples,\
                    accuracyAnnotation,\
                    nbAnnotatedSamples,\
                    totNumberSamples,\
                    initNbAnnotateSamples = exp.propagateLabels(propagation_mode,\
                                                                 percentageLabelsKeep,
                                                                 K=K,\
                                                                 localQualThresh=localQualThresh,\
                                                                 sorted_qualities=sorted_qualities)
                    # Saving the results
                    if (localQualThresh not in results['accs']):
                        results['accs'][localQualThresh] = [accuracyAnnotation]
                        results['nbsAnnotatedSamples'][localQualThresh] = [nbAnnotatedSamples]
                        results['initNbAnnotateSamples'][localQualThresh] = [initNbAnnotateSamples]
                        results['nbTotalSamples'][localQualThresh] = [totNumberSamples]
                    else:
                        results['accs'][localQualThresh].append(accuracyAnnotation)
                        results['nbsAnnotatedSamples'][localQualThresh].append(nbAnnotatedSamples)
                        results['initNbAnnotateSamples'][localQualThresh].append(initNbAnnotateSamples)
                        results['nbTotalSamples'][localQualThresh].append(totNumberSamples)

    elif (var_to_study == 'gridSearch'):
        print("\n=======>Variable studied: {}".format(var_to_study))
        if (propagation_mode != 'propLocalQual'):
            raise ValueError("If var_to_study is 'gridSearch', then mode has to be 'propLocalQual'" )
        else:

            K_vals = [i for i in range(21)]
            localQualThresh_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            percentageLabelsKeep = 0.1
            results = {
                        'K': K_vals.copy(),
                        'accs': {},
                        'nbsAnnotatedSamples': {},
                        'localQualThresh': localQualThresh_vals.copy(),
                        'percentageLabelsKeep': percentageLabelsKeep,
                        'initNbAnnotateSamples': {},
                        'nbTotalSamples': {}
                        }
            for K in K_vals:
                for localQualThresh in localQualThresh_vals:
                    print("\t===> Value of (K, localQualThresh) = ({}, {})".format(K))
                    print("\n=======> Doing computation for K = {} and localQualThresh = {}".format(K, localQualThresh))
                    startTime = time()
                    for repetition_number in range(nbRepetitions):
                        print("=======> Repetition n°{} <=======".format(repetition_number))
                        # Doing the propagation
                        newAnnotatedSamples,\
                        accuracyAnnotation,\
                        nbAnnotatedSamples,\
                        totNumberSamples,\
                        initNbAnnotateSamples = exp.propagateLabels(propagation_mode,\
                                                                     percentageLabelsKeep,
                                                                     K=K,\
                                                                     localQualThresh=localQualThresh,\
                                                                     sorted_qualities=sorted_qualities)

                        # Saving the results
                        couple_K_tau = (K, localQualThresh)
                        if (couple_K_tau not in results['accs']):
                            results['accs'][couple_K_tau] = [accuracyAnnotation]
                            results['nbsAnnotatedSamples'][couple_K_tau] = [nbAnnotatedSamples]
                            results['initNbAnnotateSamples'][couple_K_tau] = [initNbAnnotateSamples]
                            results['nbTotalSamples'][couple_K_tau] = [totNumberSamples]
                        else:
                            results['accs'][couple_K_tau].append(accuracyAnnotation)
                            results['nbsAnnotatedSamples'][couple_K_tau].append(nbAnnotatedSamples)
                            results['initNbAnnotateSamples'][couple_K_tau].append(initNbAnnotateSamples)
                            results['nbTotalSamples'][couple_K_tau].append(totNumberSamples)
                    endTime = time()
                    print("=======> Finishing computation for K = {} and localQualThresh = {} (computation done in {} s for {} repetitions)\n".format(K, localQualThresh, endTime - startTime, nbRepetitions))

    else:
        raise NotImplementedErrror("var_to_study {} is not supported yet".format(var_to_study))

    # Plotting the results
    plotResults(results, var_to_study)

    # Saving the results
    if (len(results) > 0):
        # Creating the folders to store the results
        inc = 0
        results_folder = projections_folder + '/LabelPropResults/'
        if not os.path.isdir(results_folder):
            os.mkdir(results_folder)

        # Writing the results
        fileName = results_folder + exp_ID
        if (propagation_mode.lower() == 'proplocalqual'):
            fileName += '_propMode-{}_var-to-study-{}_sorted-qualities-{}'.format(propagation_mode, var_to_study, str(sorted_qualities))
            if (var_to_study.lower() != 'localqualthresh') and (var_to_study.lower() != 'gridSearch'):
                fileName += '_localQualThreshod-{}'.format(localQualThresh)
            if (var_to_study.lower() != 'k') and (var_to_study.lower() != 'gridSearch'):
                fileName += '_K-{}'.format(K)
            if (var_to_study.lower() != 'percentagelabelskeep'):
                fileName += '_percentageLabelsKeep-{}'.format(percentageLabelsKeep)

        elif (propagation_mode.lower() == 'classicalprop'):
            fileName += '_propMode-{}_var-to-study-{}'.format(propagation_mode, var_to_study, str(sorted_qualities))
            if (var_to_study.lower() != 'k'):
                fileName += '_K-{}'.format(K)
            if (var_to_study.lower() != 'percentagelabelskeep'):
                fileName += '_percentageLabelsKeep-{}'.format(percentageLabelsKeep)

        elif (propagation_mode.lower() == 'opf-semi'):
            fileName += '_propMode-{}'.format(propagation_mode)

        inc = 0
        while (os.path.isfile(fileName+ '_' + str(inc) + '.pth')):
            inc += 1
        fileName = fileName + '_' + str(inc) + '.pth'

        # Creating the pickle file
        print("Saving the results of the experiment")
        with open(fileName, "wb") as fp:   #Pickling
            pickle.dump(results, fp)
        print("The results of the experiment were saved at: {}".format(fileName))


if __name__=='__main__':
    main()
