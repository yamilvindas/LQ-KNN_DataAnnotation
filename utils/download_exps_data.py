#!/usr/bin/env python3
"""
    This code downloads some useful data to run some experiments. It avoids
    to relaunch all the computations so different blocks can be tested
    independetly and relatively quickly.
"""
import os
import requests


def download_file(file_url, dir_store, file_name):
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
    """
    # Searching if the directory name exists
    if os.path.exists(dir_store+'/'+file_name):
        print("The file {} already exists".format(dir_store+'/'+file_name))
    else:
        # Downloading the file
        file = requests.get(file_url)
        open(dir_store+'/'+file_name, 'wb').write(file.content)
        print("File downloaded successfully and stored in {}".format(dir_store+'/'+file_name))


def download_dim_red_data():
    """
        Downloads the data needed to run the examples/dim_red.py experiment
        without any argument
    """
    # Creating the folder that wil contain the data  (if it does not exists)
    if (not os.path.exists('../models/MNIST_Example_0/')):
        os.mkdir('../models/MNIST_Example_0/')
    if (not os.path.exists('../models/MNIST_Example_0/CompressedRepresentations/')):
        os.mkdir('../models/MNIST_Example_0/CompressedRepresentations/')
    if (not os.path.exists('../models/MNIST_Example_0/Model/')):
        os.mkdir('../models/MNIST_Example_0/Model/')


    # Downloading the trained auto-encoder
    # Model
    download_file(
                    file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/models/MNIST_Example_0/Model/metrics.hdf5',\
                    dir_store='../models/MNIST_Example_0/Model/',\
                    file_name='metrics.hdf5'
                )
    # Metrics
    download_file(
                    file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/models/MNIST_Example_0/Model/model.pth',\
                    dir_store='../models/MNIST_Example_0/Model/',\
                    file_name='model.pth'
                )

    # Downloading the compresed representations in the auto-encoder lantet space
    download_file(
                    file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/models/MNIST_Example_0/CompressedRepresentations/training_representations.pth',\
                    dir_store='../models/MNIST_Example_0/CompressedRepresentations/',\
                    file_name='training_representations.pth'
                )

def download_label_propagation_data():
    """
        Downloads the data needed to run the examples/label_propagation.py experiment
        without any argument
    """
    # Downloading the data used in the steps before (i.e. the trained auto-encoder,
    # the compressed representations in the auto-encoder latent space)
    download_dim_red_data()

    # Creating the directories for the dimensionality reduction results that
    # are going to be downloaded
    if (not os.path.exists('../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/')):
        os.mkdir('../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/')
    projections_folders = [
                            'EmbeddedRepresentations_perp10_lr1000_earlyEx500_dim2_0',\
                            'EmbeddedRepresentations_perp30_lr100_earlyEx250_dim2_0',\
                            'EmbeddedRepresentations_perp30_lr1000_earlyEx50_dim2_0'
                          ]
    for projection_folder in projections_folders:
        if (not os.path.exists('../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/'+projection_folder)):
            os.mkdir('../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/'+projection_folder)

    # Dowloading the results json files containing the best, middle and worst
    # selected projections by out method
    download_file(
                    file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/resultsProjectionSelection.json',\
                    dir_store='../models/MNIST_Example_0//Projections_Example-Dim-Reduction_0/',\
                    file_name='resultsProjectionSelection.json'
                )

    # For each projection folder, downloading the different files
    files_to_download = ['corankingMatrix_0.pth', 'dataSplits_0.pth',\
    'images_0.pth', 'labels_0.pth', 'localQuality_ks10_kt10_0.pth',\
    'quality_0.pth', 'ranksR_0.pth', 'ranksRho_0.pth', 'representations_0.pth',\
    'tsne_params.json']
    for projection_folder in projections_folders:
        for file_to_download in files_to_download:
            download_file(
                            file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/{}/{}'.format(projection_folder, file_to_download),\
                            dir_store='../models/MNIST_Example_0//Projections_Example-Dim-Reduction_0/'+projection_folder,\
                            file_name=file_to_download
                        )



def download_label_propagation_with_classification_data():
    """
        Downloads the data needed to run the examples/label_propagation_with_classification.py
        experiment without any argument
    """
    # It downloads the same files as label_propagation_data()
    download_label_propagation_data()

def download_label_propagation_results_classification_data():
    """
        Downlaod the necessary data to plot the classification results using the
        semi-automatically labeled datasets shown in the paper for MNIST
    """
    # Creting the needed directories
    if (not os.path.exists('../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/')):
        os.mkdir('../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/')
    if (not os.path.exists('../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/EmbeddedRepresentations_perp30_lr1000_earlyEx50_dim2_0/')):
        os.mkdir('../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/EmbeddedRepresentations_perp30_lr1000_earlyEx50_dim2_0/')
    if (not os.path.exists('../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/EmbeddedRepresentations_perp30_lr1000_earlyEx50_dim2_0/ClassificationResults/')):
        os.mkdir('../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/EmbeddedRepresentations_perp30_lr1000_earlyEx50_dim2_0/ClassificationResults/')

    # Downloading the files
    files_to_download = ['LQ-kNN_CE_0.pth', 'LQ-kNN_GCE_0.pth', 'NoProp_CE_0.pth',\
    'NoProp_GCE_0.pth', 'Std-kNN_CE_0.pth', 'Std-kNN_GCE_0.pth']
    for file_to_download in files_to_download:
        download_file(
                        file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/EmbeddedRepresentations_perp30_lr1000_earlyEx50_dim2_0/ClassificationResults/'+file_to_download,\
                        dir_store='../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/EmbeddedRepresentations_perp30_lr1000_earlyEx50_dim2_0/ClassificationResults/',\
                        file_name=file_to_download
                    )


def download_label_propagation_results_data():
    """
        Download the necessary data to plot the label propagation results on MNIST
        shown in the paper.
    """
    # Creting the needed directories
    if (not os.path.exists('../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/')):
        os.mkdir('../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/')
    if (not os.path.exists('../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/EmbeddedRepresentations_perp30_lr1000_earlyEx50_dim2_0/')):
        os.mkdir('../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/EmbeddedRepresentations_perp30_lr1000_earlyEx50_dim2_0/')
    if (not os.path.exists('../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/EmbeddedRepresentations_perp30_lr1000_earlyEx50_dim2_0/LabelPropResults/')):
        os.mkdir('../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/EmbeddedRepresentations_perp30_lr1000_earlyEx50_dim2_0/LabelPropResults/')

    # Downloading the files
    files_to_download = ['TODO!!!!!']
    for file_to_download in files_to_download:
        download_file(
                        file_url='https://www.creatis.insa-lyon.fr/~vindas/LQ-KNN_DataAnnotation/models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/EmbeddedRepresentations_perp30_lr1000_earlyEx50_dim2_0/LabelPropResults/'+file_to_download,\
                        dir_store='../models/MNIST_Example_0/Projections_Example-Dim-Reduction_0/EmbeddedRepresentations_perp30_lr1000_earlyEx50_dim2_0/LabelPropResults/',\
                        file_name=file_to_download
                    )
