#!/usr/bin/env python3
"""
    This code implements some evaluation metrics for dimensionality reduction
    mainly based on the co-ranking matrix. All is based on the library
    coranking.
    WARNING TO TAKE INTO ACCOUNT: we do not consider 0 as a rank (because if we
    have a sample A, the only sample that has a rank of 0 with respect to A is
    A itself!). This means that when we compute the Lueks quality for k_s = 1,
    we have a Lueks quality of 0 (because if k_s = 0, the only samples that are
    interesting for us are the ones having rank 0, which there are not)!!! Another
    way to see it is that, if k_s is 1, then the region of interest in the
    co-ranking matrix are the ones having a ranking of 0, which does not appear
    in the co-ranking matrix. If we focus only on the samples having as rank 0
    in the high dimensional space and in the low dimensional space (that is
    what k_s = 1 means), then we will have exactly N samples (total number
    of samples) respecting this property, so if we do not set the case of the
    quality when k_s = 1 to 0, the real value with the formula in the paper is
    1, which is logical but has no practical use.

    Options:
        *--projections_folder: Folder containing the files with the high and low dimensional data.
        This folder is usually located in ../models/MNIST_EXP_ID/Projections_ExpID/EmbeddedRepresentations_perp{}_lr{}_earlyEx{}_dim{}_ID/
        *--latent_space_repr: Path to a file contained the compressed representations obtained by an AE model.
        If this file exists, it is usually located in the folder ../models/MNIST_EXP_ID/CompressedRepresentations/
        *--coranking_matrix_file: If a file containing the coranking matrix exists, it can be used to accelerate computation.
        If this file exists, it is usually located in the folder ../models/MNIST_EXP_ID/Projections_ExpID/EmbeddedRepresentations_perp{}_lr{}_earlyEx{}_dim{}_ID/
        *--ks: Size of the neighborhood used to compute the local quality
        *--kt: Rank error tolerance used to compute the local quality
        *--trustworthiness: True if want to compute the trustworthiness
        *--continuity: True if want to compute the continuity
        *--lcmc: True if want to compute the lcmc
        *--quality: True if want to compute the quality
        *--quality_lueks: True if want to compute the qualityLueks
        *--local_quality: True if want to compute the localQuality

    Files generated:
        * If projections_folder is used, then the coranking matrix is calculated and
        stored in projections_folder.
        * For all cases, the trustworthiness, continuity and lcmc are computed
        and stored in projections_folder
"""
import os
import argparse
import coranking
from coranking.metrics import trustworthiness, continuity, LCMC
from time import time
from matplotlib import image
import pickle
import numpy as np
from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from scipy.spatial import distance


def loadHighAndLowData(folder_embRepr, latent_space_repr=None):
    """
        Loads the high dimensional data and the embedded data on numpy arrays.

        Arguments:
        ----------
        folder_embRepr: str
            Path to the folder containing the different files where we can find
            the high dimensional data and its embedded representation
        latent_space_repr: str
            Path to a file contained the compressed representations obtained by an AE model.
            WARNING: If not specified the code uses the original input space as
            high dimensional space.

        Returns:
        --------
        highDimData: numpy.array
            High Dimensional data
        lowDimData: numpy.array
            Low dimensional data. It is not necessary of low dimension, but its
            dimension is lower thant the original data dimension
    """
    # Putting the path under the right format
    if (folder_embRepr[-1] != "/"):
        folder_embRepr += "/"

    # Getting the different paths names
    pointFile = folder_embRepr + 'representations_0.pth'
    imagesFile = folder_embRepr + 'images_0.pth'

    # Loadinf the data
    with open(pointFile, "rb") as fp:   # Unpickling
        lowDimData = pickle.load(fp)

    # For the high dimensional data
    if (latent_space_repr is not None):
        print("======> Using the AE latent feature space as high dimensional space")
        latent_space_reprFile = ''
        if (latent_space_repr is not None):
            latent_space_reprFile = latent_space_repr
        else:
            latent_space_reprFile = folder_embRepr+'AE_representations_0.pth'
        print("Using {} file as latent space file".format(latent_space_reprFile))

        with open(latent_space_reprFile, "rb") as fp:   # Unpickling
            highDimData = pickle.load(fp)

        highDimData = rearangeData(highDimData)
        highDimData = highDimData[-1]
        highDimData = np.array(highDimData['compressed_representation'])
        highDimDataShape = highDimData.shape
        # highDimData = np.reshape(highDimData, (highDimDataShape[0], highDimDataShape[1]*highDimDataShape[2]*highDimDataShape[3]))
    else:
        print("=======> Using original image space as high dimensional space")
        with open(imagesFile, "rb") as fp:   # Unpickling
            images = pickle.load(fp)
        highDimData = np.array(images)
        highDimDataShape = highDimData.shape
        highDimData = np.reshape(highDimData, (highDimDataShape[0], highDimDataShape[1]*highDimDataShape[2]*highDimDataShape[3]))

    return highDimData, lowDimData

################################################################################
################################ Classica quality ##############################
################################################################################
def quality(Q, K):
    """
        Implementation of the first quality measure of a dimensionality reduction
        technique introduced in the paper of Lueks et al. (2011)

        Arguments:
        ----------
        Q: numpy.array:
            Co-ranking matrix.
        K: int
            Controls the region of interest for the ranks to take into account
            AND the tolerated rank errors

        Returns:
        --------
        q: float
            Quality measure for the dimensionality reduction technique
    """
    N = Q.shape[0]+1
    q = 0
    for i in range(K):
        for j in range(K):
            q += Q[i,j]
    q = (1./(K*N))*q
    return q

################################################################################
############################ Quality Lueks et al. ##############################
################################################################################
def quality_lueks(Q, k_s, k_t):
    """
        Implementation of the second quality measure of a dimensionality reduction
        technique introduced in the paper of Lueks et al. (2011)

        Arguments:
        ----------
        Q: numpy.array:
            Co-ranking matrix.
        k_t: int
            Controls which rank errors are tolerated
        k_s: int
            Controls the region of interest for the ranks (out of the ROI, the
            rangs are rejected)

        Returns:
        --------
        q_l: float
            Quality measure for the dimensionality reduction technique
    """
    N = Q.shape[0] + 1
    q_l = 0
    # Defining the two functions used in the original paper
    w_s = lambda rho, r, k: 0 if (rho > k and r > k) else 1
    w_t = lambda rho, r, k: 0 if abs(rho-r)>k else 1
    for i in range(N-1):
        for j in range(N-1):
            # WARNING: AS WE ARE MANIPULATING ARRAYS WHERE THE INDICES START FROM 0,
            # THE FORMULA OF THE PAPER HAS TO BE MODIFIED AND WE HAVE TO SUBSTRACT
            # MINUS 1 TO THE HYPER-PARAMETERS k_s and k_t
            q_l += w_s(i+1,j+1,k_s-1)*w_t(i+1,j+1,k_t-1)*Q[i][j]
    q_l = (1/(k_s*N))*q_l

    if (q_l > 1):
        print("Quality obtained ({}) superior to 1 for k_s = {} and k_t = {} ".format(q_l, k_s, k_t))
        exit()

    return q_l



################################################################################
################################ Local Quality #################################
################################################################################
def computeAllRanks(i, E):
    """
        Computes the ranks all of the points of E with respect to the point
        (of index) i

        Arguments:
        ----------
        i: int
            Sample (index) of reference to compute the ranks of the other points (all
            the ranks computed are done with respect to this point)
        E: list or numpy.array
            Space of all the points where i belongs

        Returns:
        --------
        ranks: dict
            Dictionary where the key is the index of the sample in the space E
            and the value is the rank of the sample with respect to the sample of
            reference i. In other words is a dict containing the ranks of all
            the sample in E with respect to the sample i

    """
    # Selecting the reference sample (we are going to compute the ranks of the
    # other samples with respect to this one)
    x_i = np.reshape(E[i,:], (1, E.shape[1]))

    # Computing the distances between this sample and the others
    distances = distance.cdist(x_i, E, 'euclidean')

    # Sorting the samples to get them by increasing order of distances (so
    # the index of the distance is the rank of the sample)
    list_distances = list(np.reshape(distances, distances.shape[1]))
    # print("Original distances: ", list_distances)
    # We sort the "enumerate" of the distances because it allow us to keep the
    # original index of the sample, that corresonds to the identifier of the sample
    ordered_distances = sorted(enumerate(list_distances), key=lambda i: i[1])
    # print("Ordered distances: ", ordered_distances)

    # Converting the list into a dict
    ranks = {}
    for rank in range(len(ordered_distances)):
        sampleID = ordered_distances[rank][0]
        ranks[sampleID] = rank
    # print("Dict of the ranks: ", ranks)

    return ranks


def local_quality(i, ranksRho, ranksR, k_s, k_t):
    """
        Function that computes the local quality of i when using a dimensionality
        reduction technique.

        Arguments:
        ----------
        i: int
            Sample from which we want to compute the local quality
        ranksRho: dict
            Dict containing the ranks of all the sample in the original space
            with respect to the sample i
        ranksR: dict
            Dict containing the ranks of all the sample in the embedded space
            with respect to the sample i
        k_s: int
            Controls the region of interest for the ranks (out of the ROI, the
            rangs are rejected)
        k_t: int
            Controls which rank errors are tolerated

        Returns:
        --------
        q_local: float
            Local quality of the point i when using a particular dimensionality
            reduction technique
    """
    # Number of samples
    N = len(ranksRho)

    # Defining the two functions used in the original paper
    # Here we DO NOT HAVE to adjust the K by substracting one because we do not
    # use the co-ranking matrix but the ranks directly (the co-ranking matrix
    # is represented as an array where the index go from 0 to N-2 and not from 1 to N-1)
    w_s = lambda rho, r, k: 0 if (rho > k and r > k) else 1
    w_t = lambda rho, r, k: 0 if abs(rho-r)>k else 1

    # Computing the local quality
    q_local = 0
    for j in range(N):
        rho_ij = ranksRho[i][j]
        r_ij = ranksR[i][j]
        rho_ji = ranksRho[j][i]
        r_ji = ranksR[j][i]
        q_local += (w_s(rho_ij, r_ij, k_s)*w_t(rho_ij, r_ij, k_t)+w_s(rho_ji, r_ji, k_s)*w_t(rho_ji, r_ji, k_t))
    q_local = (1/(2*k_s*N))*q_local

    return q_local


def rearangeData(data):
    """
        Function that puts the data of the list of points under the right format:

        Arguments:
        ----------
        data: list
            List where each element is a dict that correspond to one epoch of the training
            process. Each dict contains two keys:
                - 'compressed_representation': List where each element is the
                output of a batch for the given epoch.
                - 'label': List where each element are the labels of the output of a
                batch for a given epoch

        Returns:
        --------
        neData: list
            List containing the data points to plot. The elements of the list
            are dictionaries with two keys:
                - 'compressed_representation': List where each element is one
                output for that epoch
                - 'label': List where each element is the label of the corresponding
                output point in the output list
    """
    newData = []
    for epochData in data:
        newOutput, newLabel, newDataSplits = [], [], []
        for labelLists in epochData['label']:
            nbSamplesBatch = len(labelLists)# It's the same as len(labelLists[1]), ..,. len(labelLists[9])
            for i in range(nbSamplesBatch):
                newLabel.append(int(labelLists[i]))
        for outputLists in epochData['compressed_representation']:
            for output in outputLists:
                newOutput.append(output.cpu().detach().numpy())
        if ('data_split' in epochData):
            for dataSplitLists in epochData['data_split']:
                for dataSplit in dataSplitLists:
                    newDataSplits.append(dataSplit)
        newData.append({'compressed_representation': newOutput, 'label': newLabel, 'data_split':newDataSplits})
    return newData


def main():
    # =========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("--projections_folder", required=False, help="Folder containing the files with the high and low dimensional data", type=str)
    ap.add_argument("--coranking_matrix_file", required=False, help="If a file containing the coranking matrix exists, it can be used to accelerate computation", type=str)
    ap.add_argument("--latent_space_repr", required=False, help="Path to a file contained the compressed representations obtained by an AE model", type=str)
    ap.add_argument("--ks", default=10, help="Size of the neighborhood used to compute the local quality", type=int)
    ap.add_argument("--kt", default=10, help="Rank error tolerance used to compute the local quality", type=int)
    ap.add_argument("--trustworthiness", default="False", help="True if want to compute the trustworthiness", type=str)
    ap.add_argument("--continuity", default="False", help="True if want to compute the continuity", type=str)
    ap.add_argument("--lcmc", default="False", help="True if want to compute the lcmc", type=str)
    ap.add_argument("--quality", default="False", help="True if want to compute the quality", type=str)
    ap.add_argument("--quality_lueks", default="False", help="True if want to compute the qualityLueks", type=str)
    ap.add_argument("--local_quality", default="False", help="True if want to compute the localQuality", type=str)
    args = vars(ap.parse_args())

    # Parameters
    projections_folder = args['projections_folder']
    if (projections_folder[-1] != '/'):
        projections_folder += '/'
    corankMatrixFileName = args['coranking_matrix_file']
    if (projections_folder is None) and (corankMatrixFileName is None):
        raise ValueError("One of the two options --projections_folder or --coranking_matrix_file must be used")
    latent_space_repr = args['latent_space_repr']
    if (latent_space_repr.lower() == 'none'):
        latent_space_repr = None
    ks, kt = args['ks'], args['kt']
    compute_trustworthiness = args["trustworthiness"]
    compute_continuity = args["continuity"]
    compute_lcmc = args["lcmc"]
    compute_quality = args["quality"]
    compute_qualityLueks = args["quality_lueks"]
    compute_localQuality = args["local_quality"]

    #==========================================================================#
    # Loading the data
    if (projections_folder is not None):
        # Loading the high dim and low dim data
        highDimData, lowDimData = loadHighAndLowData(projections_folder, latent_space_repr)
        print("Original dimension ", highDimData.shape[1])
        print("New dimension ", lowDimData.shape[1])

        # Number of samples
        N = highDimData.shape[0]

    if (corankMatrixFileName is None):
        if (os.path.exists(projections_folder+'/corankingMatrix_0.pth')):
            print("WARNING: Using existing coranking matrix file: {}".format(projections_folder+'/corankingMatrix_0.pth'))
            # Loading the high and low dimensional data (required to compute local qualities)
            highDimData, lowDimData = loadHighAndLowData(projections_folder, latent_space_repr)
            # Loading the coranking matrix
            Q = None
            with open(projections_folder+'/corankingMatrix_0.pth', "rb") as fp:   # Unpickling
                Q = pickle.load(fp)
            N = Q.shape[0] + 1
        else:
            # Computing the coranking matrix
            startTime = time()
            Q = coranking.coranking_matrix(highDimData, lowDimData)
            endTime = time()
            print("Time needed to compute the co-ranking matrix: {} s".format(endTime-startTime))
            print("Shape of the co-ranking matrix : {}\n".format(Q.shape))

            # Saving the co-ranking matrix
            corankMatrixFileName = projections_folder + "/corankingMatrix"
            inc = 0
            while os.path.isfile(corankMatrixFileName+'_'+str(inc)+'.pth'):
                inc +=1
            corankMatrixFileName += '_'+str(inc)+'.pth'
            with open(corankMatrixFileName, "wb") as fp:   #Pickling
                pickle.dump(Q, fp)
    else:
        # Getting the path of the folder containing the file
        projections_folder = '/'.join(corankMatrixFileName.split('/')[:-1])
        # Loading the high and low dimensional data (required to compute local qualities)
        highDimData, lowDimData = loadHighAndLowData(projections_folder, latent_space_repr)
        # Loading the coranking matrix
        Q = None
        with open(corankMatrixFileName, "rb") as fp:   # Unpickling
            Q = pickle.load(fp)
        N = Q.shape[0] + 1

    # Computing some evaluation metrics
    # Trustworthiness computation
    if (compute_trustworthiness.lower() == 'true'):
        startTime = time()
        trust = trustworthiness(Q, min_k=1, max_k=100)
        endTime = time()
        print("Time needed to compute the trustworthiness: {} s".format(endTime-startTime))
        # Saving the trustworthiness
        trustworthinessFileName = projections_folder + "/trustworthiness"
        inc = 0
        while os.path.isfile(trustworthinessFileName+'_'+str(inc)+'.pth'):
            inc +=1
        trustworthinessFileName += '_'+str(inc)+'.pth'
        with open(trustworthinessFileName, "wb") as fp:   #Pickling
            pickle.dump(trust, fp)

    # Continuity computation
    if (compute_continuity.lower() == 'true'):
        startTime = time()
        cont = continuity(Q, min_k=1, max_k=100)
        endTime = time()
        print("Time needed to compute the continuity: {} s".format(endTime-startTime))
        # Saving the continuity
        continuityFileName = projections_folder + "/continuity"
        inc = 0
        while os.path.isfile(continuityFileName+'_'+str(inc)+'.pth'):
            inc +=1
        continuityFileName += '_'+str(inc)+'.pth'
        with open(continuityFileName, "wb") as fp:   #Pickling
            pickle.dump(cont, fp)

    # LCMC computation
    if (compute_lcmc.lower() == 'true'):
        startTime = time()
        lcmc = LCMC(Q, min_k=1, max_k=100)
        endTime = time()
        print("Time needed to compute the LCMC: {} s".format(endTime-startTime))
        # Saving the LCMC
        lcmcFileName = projections_folder + "/lcmc"
        inc = 0
        while os.path.isfile(lcmcFileName+'_'+str(inc)+'.pth'):
            inc +=1
        lcmcFileName += '_'+str(inc)+'.pth'
        with open(lcmcFileName, "wb") as fp:   #Pickling
            pickle.dump(lcmc, fp)

    # Quality computation
    if (compute_quality.lower() == 'true'):
        startTime = time()
        qualityVals = []
        for K in range(1, 100):
            qualityVals.append(quality(Q, K))
        endTime = time()
        print("Time needed to compute the quality: {} s".format(endTime-startTime))
        # Saving the quality
        qualityFileName = projections_folder + "/quality"
        inc = 0
        while os.path.isfile(qualityFileName+'_'+str(inc)+'.pth'):
            inc +=1
        qualityFileName += '_'+str(inc)+'.pth'
        with open(qualityFileName, "wb") as fp:   #Pickling
            pickle.dump(qualityVals, fp)

    # Quality Lueks et al. computation
    if (compute_qualityLueks.lower() == 'true'):
        startTime = time()
        qualityLueksVals = []
        # k_s_vals = [i for i in range(1, 20)]
        k_s_vals = [i for i in [5, 10, 15, 20]]
        # k_t_vals = [i for i in range(1, 20)]
        k_t_vals = [i for i in [5, 10, 15, 20]]
        for k_s in tqdm(k_s_vals):
            tmpQualityVals = []
            for k_t in k_t_vals:
                q_l = quality_lueks(Q, k_s, k_t)
                tmpQualityVals.append(q_l)
            qualityLueksVals.append(tmpQualityVals)
        qualityLueksVals = np.array(qualityLueksVals)
        endTime = time()
        print("Time needed to compute the quality from Lueks et al.: {} s".format(endTime-startTime))
        toSaveQualityLueks = {"qualityVals":qualityLueksVals, "k_s_vals": k_s_vals, "k_t_vals":k_t_vals}
        # Saving the quality of Lueks et al.
        qualityLueksFileName = projections_folder + "/qualityLueks"
        inc = 0
        while os.path.isfile(qualityLueksFileName+'_'+str(inc)+'.pth'):
            inc +=1
        qualityLueksFileName += '_'+str(inc)+'.pth'
        with open(qualityLueksFileName, "wb") as fp:   #Pickling
            pickle.dump(toSaveQualityLueks, fp)


    # Computing the local quality of each point
    # Precomputing all the ranks
    if (os.path.exists(projections_folder+"/ranksRho_0.pth") and os.path.exists(projections_folder+"/ranksR_0.pth")):
        print("WARNING: Using existing pre-computed files for the ranks: {} and {}".format(projections_folder+"/ranksRho_0.pth", projections_folder+"/ranksR_0.pth"))
        # Loading the ranks rho
        rho = None
        with open(projections_folder+"/ranksRho_0.pth", "rb") as fp:   # Unpickling
            rho = pickle.load(fp)
        # Loading the ranks r
        r = None
        with open(projections_folder+"/ranksR_0.pth", "rb") as fp:   # Unpickling
            r = pickle.load(fp)
    else:
        startTime = time()
        rho, r = [], []
        for i in range(N):
            ranksRho = computeAllRanks(i, highDimData)
            ranksR = computeAllRanks(i, lowDimData)
            rho.append(ranksRho)
            r.append(ranksR)
        endTime = time()
        print("Time needed to compute the ranks of all the samples with respect to the other samples: {} s".format(endTime-startTime))
        # Saving the ranks
        # Ranks Rho
        ranksRhoFileName = projections_folder + "/ranksRho"
        inc = 0
        while os.path.isfile(ranksRhoFileName+'_'+str(inc)+'.pth'):
            inc +=1
        ranksRhoFileName += '_'+str(inc)+'.pth'
        with open(ranksRhoFileName, "wb") as fp:   #Pickling
            pickle.dump(rho, fp)
        # Ranks R
        ranksRFileName = projections_folder + "/ranksR"
        inc = 0
        while os.path.isfile(ranksRFileName+'_'+str(inc)+'.pth'):
            inc +=1
        ranksRFileName += '_'+str(inc)+'.pth'
        with open(ranksRFileName, "wb") as fp:   #Pickling
            pickle.dump(r, fp)

    # Computing the local qualities
    if (compute_localQuality.lower() == 'true'):
        # Checking if the local qualities for the chosen values of ks and kt
        # already exists
        if (os.path.exists(projections_folder + "/localQuality_ks{}_kt{}_0.pth".format(ks, kt))):
            print("WARNING: Computation of the local qualities not done because there is already a file with those local qualitys at {}".format(projections_folder + "/localQuality_ks{}_kt{}_0.pth".format(ks, kt)))
        else:
            local_qualities = []
            # ks, kt = 5, 15
            # ks, kt = 10, 10
            for i in tqdm(range(N)):
                quality_i = local_quality(i, rho, r, ks, kt)
                local_qualities.append(quality_i)
            print("Number of local qualities computed: {}. Number of points: {}".format(len(local_qualities), N))
            # Saving the local qualities
            localQualityFileName = projections_folder + "/localQuality_ks{}_kt{}".format(ks, kt)
            inc = 0
            while os.path.isfile(localQualityFileName+'_'+str(inc)+'.pth'):
                inc +=1
            localQualityFileName += '_'+str(inc)+'.pth'
            with open(localQualityFileName, "wb") as fp:   #Pickling
                pickle.dump(local_qualities, fp)

#==============================================================================#
#==============================================================================#
if __name__=="__main__":
    main()
