#!/usr/bin/env python3
"""
    This code downloads some useful data to run some experiments. It avoids
    to relaunch all the computations so different blocks can be tested
    independetly and relatively quickly.
"""
import os
import requests
from tqdm import tqdm

def download_file(file_url, dir_store, file_name, verbose=True):
    """
        Download a file from the given url and stores it in the given directory
        with the given name

        Arguments:
        ----------
        file_url: str
            Url to the file that we want to download
        dir_store: str
            Path of the directory where the downloaded file should be stored.
        file_name: str
            Name of the file when download it
        verbose: bool
            True if you want to print some information about the requested file
    """
    # Searching if the directory name exists
    if os.path.exists(dir_store+'/'+file_name) and verbose:
        print("The file {} already exists".format(dir_store+'/'+file_name))
    else:
        # Downloading the file
        file = requests.get(file_url)
        open(dir_store+'/'+file_name, 'wb').write(file.content)
        if (verbose):
            print("File downloaded successfully and stored in {}".format(dir_store+'/'+file_name))

def download_organc_mnist():
    """
        Downloads the OrganCMNIST dataset necessary to run the different
        experiments.
    """
    if (not os.path.exists('../datasets/organcmnist_0/')):
        print("\n=======> Starting download of theOrganCMNIST dataset")
        # Creating the directories
        os.mkdir('../datasets/organcmnist_0/')
        os.mkdir('../datasets/organcmnist_0/train/')
        os.mkdir('../datasets/organcmnist_0/val/')
        os.mkdir('../datasets/organcmnist_0/test/')

        # Files to download
        train_files = ['train_labels.csv'] + ['train_image_{}.png'.format(i) for i in range(13000)]
        val_files = ['val_labels.csv'] + ['val_image_{}.png'.format(i) for i in range(2392)]
        test_files = ['test_labels.csv'] + ['test_image_{}.png'.format(i) for i in range(8268)]
        files_to_download = {
                                'train': train_files,
                                'val': val_files,
                                'test': test_files
                            }

        # Downloading the different files
        # Data description file
        download_file(
                        file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/datasets/organcmnist_0/data.hdf5',\
                        dir_store='../datasets/organcmnist_0/',\
                        file_name='data.hdf5',\
                        verbose=False
                    )
        # Training, validation and testing files
        splits = ['train', 'val', 'test']
        for split in tqdm(splits):
            for file_to_download in files_to_download[split]:
                download_file(
                                file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/datasets/organcmnist_0/{}/{}'.format(split, file_to_download),\
                                dir_store='../datasets/organcmnist_0/{}/'.format(split),\
                                file_name=file_to_download,\
                                verbose=False
                            )
        print("=======> OrganCMNIST dataset downloaded successfully!\n")
    else:
        print("=======> OrganCMNIST dataset has alredy been downloaded!\n")



def download_dim_red_data(dataset='MNIST'):
    """
        Downloads the data needed to run the examples/dim_red.py experiment
        without any argument

        Arguments:
        ----------
        dataset: str
            Name of the dataset to use for feature extraction. Two options
            are possible: MNIST and OrganCMNIST.
    """
    # Creating the folder that wil contain the data  (if it does not exists)
    if (dataset.lower() == 'mnist'):
        dataset_name = 'MNIST'
    elif (dataset.lower() == 'organcmnist'):
        dataset_name = 'OrganCMNIST'
    if (not os.path.exists('../models/{}_Example_0/'.format(dataset_name))):
        os.mkdir('../models/{}_Example_0/'.format(dataset_name))
    if (not os.path.exists('../models/{}_Example_0/CompressedRepresentations/'.format(dataset_name))):
        os.mkdir('../models/{}_Example_0/CompressedRepresentations/'.format(dataset_name))
    if (not os.path.exists('../models/{}_Example_0/Model/'.format(dataset_name))):
        os.mkdir('../models/{}_Example_0/Model/'.format(dataset_name))



    # Downloading the trained auto-encoder
    # Model
    download_file(
                    file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/models/{}_Example_0/Model/metrics.hdf5'.format(dataset_name),\
                    dir_store='../models/{}_Example_0/Model/'.format(dataset_name),\
                    file_name='metrics.hdf5'
                )
    # Metrics
    download_file(
                    file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/models/{}_Example_0/Model/model.pth'.format(dataset_name),\
                    dir_store='../models/{}_Example_0/Model/'.format(dataset_name),\
                    file_name='model.pth'
                )

    # Downloading the compresed representations in the auto-encoder lantet space
    download_file(
                    file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/models/{}_Example_0/CompressedRepresentations/training_representations.pth'.format(dataset_name),\
                    dir_store='../models/{}_Example_0/CompressedRepresentations/'.format(dataset_name),\
                    file_name='training_representations.pth'
                )


def download_label_propagation_data(dataset='MNIST'):
    """
        Downloads the data needed to run the examples/label_propagation.py experiment
        without any argument

        Arguments:
        ----------
        dataset: str
            Name of the dataset to use for feature extraction. Two options
            are possible: MNIST and OrganCMNIST.
    """
    # Downloading the data used in the steps before (i.e. the trained auto-encoder,
    # the compressed representations in the auto-encoder latent space)
    download_dim_red_data(dataset)

    # Getting the name of the chosen dataset
    if (dataset.lower() == 'mnist'):
        dataset_name = 'MNIST'
    elif (dataset.lower() == 'organcmnist'):
        dataset_name = 'OrganCMNIST'

    # Creating the directories for the dimensionality reduction results that
    # are going to be downloaded
    if (not os.path.exists('../models/{}_Example_0/Projections_Example-Dim-Reduction_0/'.format(dataset_name))):
        os.mkdir('../models/{}_Example_0/Projections_Example-Dim-Reduction_0/'.format(dataset_name))
    if (dataset.lower() == 'mnist'):
        projections_folders = [
                                'EmbeddedRepresentations_perp10_lr10_earlyEx50_dim2_0',\
                                'EmbeddedRepresentations_perp10_lr10_earlyEx250_dim2_0',\
                                'EmbeddedRepresentations_perp10_lr10_earlyEx500_dim2_0',\
                                'EmbeddedRepresentations_perp10_lr100_earlyEx50_dim2_0',\
                                'EmbeddedRepresentations_perp10_lr100_earlyEx250_dim2_0',\
                                'EmbeddedRepresentations_perp10_lr100_earlyEx500_dim2_0',\
                                'EmbeddedRepresentations_perp10_lr1000_earlyEx50_dim2_0',\
                                'EmbeddedRepresentations_perp10_lr1000_earlyEx250_dim2_0',\
                                'EmbeddedRepresentations_perp10_lr1000_earlyEx500_dim2_0',\
                                'EmbeddedRepresentations_perp30_lr10_earlyEx50_dim2_0',\
                                'EmbeddedRepresentations_perp30_lr10_earlyEx250_dim2_0',\
                                'EmbeddedRepresentations_perp30_lr10_earlyEx500_dim2_0',\
                                'EmbeddedRepresentations_perp30_lr100_earlyEx50_dim2_0',\
                                'EmbeddedRepresentations_perp30_lr100_earlyEx250_dim2_0',\
                                'EmbeddedRepresentations_perp30_lr100_earlyEx500_dim2_0',\
                                'EmbeddedRepresentations_perp30_lr1000_earlyEx50_dim2_0',\
                                'EmbeddedRepresentations_perp30_lr1000_earlyEx250_dim2_0',\
                                'EmbeddedRepresentations_perp30_lr1000_earlyEx500_dim2_0',\
                                'EmbeddedRepresentations_perp50_lr10_earlyEx50_dim2_0',\
                                'EmbeddedRepresentations_perp50_lr10_earlyEx250_dim2_0',\
                                'EmbeddedRepresentations_perp50_lr10_earlyEx500_dim2_0',\
                                'EmbeddedRepresentations_perp50_lr100_earlyEx50_dim2_0',\
                                'EmbeddedRepresentations_perp50_lr100_earlyEx250_dim2_0',\
                                'EmbeddedRepresentations_perp50_lr100_earlyEx500_dim2_0',\
                                'EmbeddedRepresentations_perp50_lr1000_earlyEx50_dim2_0',\
                                'EmbeddedRepresentations_perp50_lr1000_earlyEx250_dim2_0',\
                                'EmbeddedRepresentations_perp50_lr1000_earlyEx500_dim2_0'
                              ]
    elif (dataset.lower() == 'organcmnist'):
        perplexities = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        early_exaggerations = [5, 10, 25, 50, 75, 100, 200, 500]
        learning_rates = [10, 50, 100, 500, 1000]
        projections_folders = ['EmbeddedRepresentations_perp{}_lr{}_earlyEx{}_dim2_0'.format(perp, lr, earlyEx) for perp in perplexities for lr in learning_rates for earlyEx in early_exaggerations]

    for projection_folder in projections_folders:
        if (not os.path.exists('../models/{}_Example_0/Projections_Example-Dim-Reduction_0/'.format(dataset_name)+projection_folder)):
            os.mkdir('../models/{}_Example_0/Projections_Example-Dim-Reduction_0/'.format(dataset_name)+projection_folder)

    # Dowloading the results json files containing the best, middle and worst
    # selected projections by out method
    download_file(
                    file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/models/{}_Example_0/Projections_Example-Dim-Reduction_0/resultsProjectionSelection.json'.format(dataset_name),\
                    dir_store='../models/{}_Example_0/Projections_Example-Dim-Reduction_0/'.format(dataset_name),\
                    file_name='resultsProjectionSelection.json'
                )

    # For each projection folder, downloading the different files
    files_to_download = ['dataSplits_0.pth', 'images_0.pth', 'labels_0.pth',\
    'localQuality_ks10_kt10_0.pth', 'representations_0.pth', 'tsne_params.json']
    for projection_folder in projections_folders:
        for file_to_download in files_to_download:
            download_file(
                            file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/models/{}_Example_0/Projections_Example-Dim-Reduction_0/{}/{}'.format(dataset_name, projection_folder, file_to_download),\
                            dir_store='../models/{}_Example_0/Projections_Example-Dim-Reduction_0/'.format(dataset_name)+projection_folder,\
                            file_name=file_to_download
                        )

def download_label_propagation_with_classification_data(dataset='MNIST'):
    """
        Downloads the data needed to run the examples/label_propagation_with_classification.py
        experiment without any argument

        Arguments:
        ----------
        dataset: str
            Name of the dataset to use for feature extraction. Two options
            are possible: MNIST and OrganCMNIST.
    """
    # It downloads the same files as label_propagation_data()
    download_label_propagation_data(dataset)

def download_label_propagation_results_classification_data(dataset='MNIST'):
    """
        Downlaod the necessary data to plot the classification results using the
        semi-automatically labeled datasets shown in the paper for MNIST

        Arguments:
        ----------
        dataset: str
            Name of the dataset to use for feature extraction. Two options
            are possible: MNIST and OrganCMNIST.
    """
    # Getting the name of the chosen dataset
    if (dataset.lower() == 'mnist'):
        dataset_name = 'MNIST'
    elif (dataset.lower() == 'organcmnist'):
        dataset_name = 'OrganCMNIST'

    # Creting the needed directories
    if (not os.path.exists('../models/{}_Example_0/'.format(dataset_name))):
        os.mkdir('../models/{}_Example_0/'.format(dataset_name))
    if (not os.path.exists('../models/{}_Example_0/Projections_Example-Dim-Reduction_0/'.format(dataset_name))):
        os.mkdir('../models/{}_Example_0/Projections_Example-Dim-Reduction_0/'.format(dataset_name))
    if (not os.path.exists('../models/{}_Example_0/Projections_Example-Dim-Reduction_0/ClassificationResults/'.format(dataset_name))):
        os.mkdir('../models/{}_Example_0/Projections_Example-Dim-Reduction_0/ClassificationResults/'.format(dataset_name))

    # Downloading the files
    if (dataset.lower() == 'mnist'):
        files_to_download = ['LQ-kNN_CE_0.pth', 'LQ-kNN_GCE_0.pth', 'NoProp_CE_0.pth',\
        'NoProp_GCE_0.pth', 'Std-kNN_CE_0.pth', 'Std-kNN_GCE_0.pth']
    elif (dataset.lower() == 'organcmnist'):
        files_to_download = ['LQ-01_K10_CE_0.pth', 'LQ-01_K10_GCE_0.pth', \
                            'NoProp_CE_0.pth', 'NoProp_GCE_0.pth', \
                            'Std-kNN_K10_CE_0.pth', 'Std-kNN_K10_GCE_0.pth']
    for file_to_download in files_to_download:
        download_file(
                        file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/models/{}_Example_0/Projections_Example-Dim-Reduction_0/ClassificationResults/'.format(dataset_name)+file_to_download,\
                        dir_store='../models/{}_Example_0/Projections_Example-Dim-Reduction_0/ClassificationResults/'.format(dataset_name),\
                        file_name=file_to_download
                    )


def download_label_propagation_results_data(dataset='MNIST'):
    """
        Download the necessary data to plot the label propagation results on MNIST
        shown in the paper.

        Arguments:
        ----------
        dataset: str
            Name of the dataset to use for feature extraction. Two options
            are possible: MNIST and OrganCMNIST.
    """
    # Getting the name of the chosen dataset
    if (dataset.lower() == 'mnist'):
        dataset_name = 'MNIST'
    elif (dataset.lower() == 'organcmnist'):
        dataset_name = 'OrganCMNIST'

    # Creting the needed directories
    if (not os.path.exists('../models/{}_Example_0/'.format(dataset_name))):
        os.mkdir('../models/{}_Example_0/'.format(dataset_name))
    if (not os.path.exists('../models/{}_Example_0/Projections_Example-Dim-Reduction_0/'.format(dataset_name))):
        os.mkdir('../models/{}_Example_0/Projections_Example-Dim-Reduction_0/'.format(dataset_name))
    if (not os.path.exists('../models/{}_Example_0/Projections_Example-Dim-Reduction_0/LabelPropResults/'.format(dataset_name))):
        os.mkdir('../models/{}_Example_0/Projections_Example-Dim-Reduction_0/LabelPropResults/'.format(dataset_name))

    # Downloading the files
    if (dataset.lower() == 'mnist'):
        files_to_download = [
                                'LQ-KNN-01_propMode-propLocalQual_var-to-study-K_sorted-qualities-True_localQualThreshod-0.1_percentageLabelsKeep-0.1_0.pth',\
                                'LQ-KNN-03_propMode-propLocalQual_var-to-study-K_sorted-qualities-True_localQualThreshod-0.3_percentageLabelsKeep-0.1_0.pth',\
                                'LQ-KNN-05_propMode-propLocalQual_var-to-study-K_sorted-qualities-True_localQualThreshod-0.5_percentageLabelsKeep-0.1_0.pth',\
                                'Std-KNN_propMode-classicalProp_var-to-study-K_percentageLabelsKeep-0.1_0.pth',\
                                'OPF-Semi_propMode-OPF-Semi_var-to-study-None_percentageLabelsKeep-0.1_0.pth'
                            ]
    elif (dataset.lower() == 'organcmnist'):
        files_to_download = [
                                'LQ-KNN-01_propMode-propLocalQual_var-to-study-K_sorted-qualities-True_localQualThreshod-0.1_percentageLabelsKeep-0.1_ProjType-Best_0.pth',\
                                'LQ-KNN-01_propMode-propLocalQual_var-to-study-K_sorted-qualities-True_localQualThreshod-0.1_percentageLabelsKeep-0.1_ProjType-Worst_0.pth',\
                                'LQ-KNN-03_propMode-propLocalQual_var-to-study-K_sorted-qualities-True_localQualThreshod-0.3_percentageLabelsKeep-0.1_ProjType-Best_0.pth',\
                                'LQ-KNN-03_propMode-propLocalQual_var-to-study-K_sorted-qualities-True_localQualThreshod-0.3_percentageLabelsKeep-0.1_ProjType-Worst_0.pth',\
                                'LQ-KNN-05_propMode-propLocalQual_var-to-study-K_sorted-qualities-True_localQualThreshod-0.5_percentageLabelsKeep-0.1_ProjType-Best_0.pth',\
                                'LQ-KNN-05_propMode-propLocalQual_var-to-study-K_sorted-qualities-True_localQualThreshod-0.5_percentageLabelsKeep-0.1_ProjType-Worst_0.pth',\
                                'OPF-Semi_propMode-OPF-Semi_var-to-study-None_percentageLabelsKeep-0.1_ProjType-Best_0.pth',\
                                'OPF-Semi_propMode-OPF-Semi_var-to-study-None_percentageLabelsKeep-0.1_ProjType-Worst_0.pth',\
                                'Std-KNN_propMode-classicalProp_var-to-study-K_percentageLabelsKeep-0.1_ProjType-Best_0.pth',\
                                'Std-KNN_propMode-classicalProp_var-to-study-K_percentageLabelsKeep-0.1_ProjType-Worst_0.pth'
                            ]
    for file_to_download in files_to_download:
        download_file(
                        file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/models/{}_Example_0/Projections_Example-Dim-Reduction_0/LabelPropResults/'.format(dataset_name)+file_to_download,\
                        dir_store='../models/{}_Example_0/Projections_Example-Dim-Reduction_0/LabelPropResults/'.format(dataset_name),\
                        file_name=file_to_download
                    )
