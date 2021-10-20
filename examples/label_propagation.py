#!/usr/bin/env python3
"""
    Evaluation of different label propagation methods on the MNIST dataset.

    Options:
    --------
        *--exp_ID: Name of the experiment
        *--folder_embRepr: Folder to the files describing the embedded data
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
        
    It stores the results in a folder named 'LabelPropResults' in the same folder
    as the embedded representations (i.e. in folder_embRepr). The name of the
    file is of the form expID_propMode-{}_var-to-study-{}.pth
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from time import time
import copy
from src.label_propagation import propagateLabels_LQKNN, propagateLabelsLocalQuality_LQKNN_withoutSort
from src.label_propagation import propagateLabels_StdKNN, propagateLabels_OPF

class Experiment(object):
    def __init__(self,\
                 exp_ID,\
                 folder_embRepr,\
                 ks,\
                 kt):
        """
            Class that compares two semi-automatic annotatio methods. The first
            one uses KNN and the local quality of samples to propagate labels
            and the second one uses only KNN

            Arguments:
            ----------
            folder_embRepr: str
                Path to a dir containing the different files needed to do the plot
                (pointsFile, labelsFile, sourcesFile, subjectsFile and pathsToImagesFile)
            ks: int
                Value of ks to choose the local quality file to use for LQ-kNN
            kt: int
                Value of kt to choose the local quality file to use for LQ-kNN
        """
        # Defining the name of the experiment
        self.expID = exp_ID

        # Defining the folder which contains the final embedded representations
        # of the original samples (in general they are 2D points) as well
        # as the local qualities for each embedded representation
        self.folderEmbRepr = folder_embRepr

        # Defining the values of ks and kt to use for the local quality
        self.ks, self.kt = ks, kt

        # Defining some attributes
        self.nbClasses = 10

        #==========================================================================#
        # Loading the data
        self.pointFile = self.folderEmbRepr + 'representations_0.pth'
        self.labelsFile = self.folderEmbRepr + 'labels_0.pth'
        self.localQualitiesFile = self.folderEmbRepr + 'localQuality_ks{}_kt{}_0.pth'.format(self.ks, self.kt)
        with open(self.pointFile, "rb") as fp:   # Unpickling
            self.embeddedRepresentations = pickle.load(fp)
        with open(self.labelsFile, "rb") as fp:   # Unpickling
            self.labels = pickle.load(fp)
        # Local qualities
        if (os.path.isfile(self.localQualitiesFile)):
            with open(self.localQualitiesFile, "rb") as fp:   # Unpickling
                self.localQualities = pickle.load(fp)
                # NORMALIZING THE LOCAL QUALITIES BY THE MAX!!!!!!
                tmpMaxLocalQuals = max(self.localQualities)
                self.localQualities = [localQual/tmpMaxLocalQuals for localQual in self.localQualities]
        else:
            self.localQualities = []
            print("\nWARNING !!! No local quality file found for ks = {} and kt = {}; impossible to propagate labels using local quality\n".format(self.ks, self.kt))


    def getSamplesToAnnotate(self, percentageLabelsKeep):
        """
            Creates a list of the status of the samples, dividing them between
            labeled an unlabeled samples. This list is build using the attribute
            self.labels; only the percentage percentageLabelsKeep of the labels
            are kept

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
        # Initiliazing the lists of labeled and unlabeled samples
        labeled_samples, unlabeled_samples = [], []

        # Creating a list of the annotation status of the samples
        # Here we are going to artificially keep only a percentage of the annotated samples
        # for each class
        nb_samples = len(self.embeddedRepresentations)


        # First let's count the number of annotated classes per class
        tot_nb_samples_per_class = {i:0 for i in range(self.nbClasses)}
        list_idx_per_class = {i:[] for i in range(self.nbClasses)}
        for idSample in range(nb_samples):
            label = self.labels[idSample]
            tot_nb_samples_per_class[label] += 1
            list_idx_per_class[label].append(idSample)

        # Generating the list of annotated status of the samples
        # Labeled samples
        nb_samples_to_label_per_class = {i:int(percentageLabelsKeep*tot_nb_samples_per_class[i]) for i in range(self.nbClasses)}
        labeled_samples_idxs = []
        for class_val in list_idx_per_class:
            idxs_labels_keep_per_class = random.sample(list_idx_per_class[class_val], nb_samples_to_label_per_class[class_val])
            for sample_id in idxs_labels_keep_per_class: # Labeled samples
                sample = {
                            'Data': self.embeddedRepresentations[sample_id],
                            'Label': self.labels[sample_id],
                            'LocalQuality': self.localQualities[sample_id]
                        }
                labeled_samples.append(sample)
                labeled_samples_idxs.append(sample_id)
        # Unlabeled samples
        for sample_id in range(nb_samples):
            if (sample_id not in labeled_samples_idxs):
                sample = {
                            'Data': self.embeddedRepresentations[sample_id],
                            'Label': self.labels[sample_id],
                            'LocalQuality': self.localQualities[sample_id]
                        }
                unlabeled_samples.append(sample)

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
    default_folder_embRepr = '../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0//EmbeddedRepresentations_perp30_lr1000_earlyEx10_dim2_0/'
    ap.add_argument('--exp_ID', default='evaluation_label_propagation_MNIST', help="Name of the experiment", type=str)
    ap.add_argument("--folder_embRepr", default=default_folder_embRepr, help="Folder to the files describing the embedded data", type=str)
    ap.add_argument('--propagation_mode', default='propLocalQual', help="Mode to propagate labels (propLocalQual or classicalProp or OPF-Semi)", type=str)
    ap.add_argument('--var_to_study', default='K', help="Variable to study (K, percentageLabelsKeep or localQualThresh)", type=str)
    ap.add_argument('--sorted_qualities', default='True', help="True if wanted to sort the samples by local quality when propagating the labels using LQ-KNN", type=str)
    ap.add_argument('--local_quality_threshold', default=0.1, help="Local quality threshold to use if propLocalQual mode is used and the variable to study is not the local quality threshold", type=float)
    ap.add_argument('--ks', default=10, help="Value of ks to choose the local quality file to use for LQ-kNN", type=str)
    ap.add_argument('--kt', default=10, help="Value of kt to choose the local quality file to use for LQ-kNN", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    exp_ID = args['exp_ID']
    folder_embRepr = args['folder_embRepr']
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


    #==========================================================================#
    # Initializing the experiment
    exp = Experiment(exp_ID, folder_embRepr, ks, kt)

    #==========================================================================#
    results = {}
    nbRepetitions = 20 # For statistical purposes
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
        results_folder = folder_embRepr + '/LabelPropResults/'
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
