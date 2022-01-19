#!/usr/bin/env python3
"""
    Computes projection metrics for different projections obtained with a
    dimensionality reduction technique and stored in a folder projections_folders.

    Options:
    --------
    --projections_folders: Folder containing the sub-folders corresponding to the different projections
    --latent_space_repr: Path to a file contained the compressed representations obtained by an AE model

    Output:
    -------
        Generates the projection metrics files in each sub-folder corresponding
        to a projection in the folder 'projections_folders'
"""
import os
import argparse
import subprocess

def main():
    # =========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument("--projections_folders", required=True, help="Folder containing the sub-folders corresponding to the different projections", type=str)
    ap.add_argument("--latent_space_repr", required=True, help="Path to a file contained the compressed representations obtained by an AE model", type=str)
    args = vars(ap.parse_args())

    # Parameters
    projections_folders = args['projections_folders']
    latent_space_repr = args['latent_space_repr']

    #==========================================================================#
    ks, kt = 10, 10
    for proj_folder in os.listdir(projections_folders):
        if ("EmbeddedRepresentations" in proj_folder) and (os.path.isdir(projections_folders + '/' + proj_folder)):
            projections_folder = projections_folders + '/' + proj_folder + '/'
            print('\n\n =======> Computing projection metrics for {}'.format(projections_folder))
            if (not os.path.isfile(projections_folder + '/' + 'localQuality_ks{}_kt{}_0.pth'.format(ks, kt))):
                with subprocess.Popen(\
                                        [
                                            'python',\
                                            './projection_metrics.py',\
                                            '--projections_folder',\
                                            projections_folder,\
                                            '--latent_space_repr',\
                                            latent_space_repr,
                                            '--ks',\
                                            str(ks),
                                            '--kt',\
                                            str(kt),
                                            '--quality_lueks',\
                                            "False",
                                            '--local_quality',\
                                            "True"
                                        ], stdout=subprocess.PIPE
                                     ) as proc:
                    # Seing if the sample was annotated
                    for line in proc.stdout:
                        line = line.decode("utf-8")
                        # print(line)
                print("\tComputation finished! \n")
            else:
                print("Local quality file for ks = {} and kt = {} already exists! \n".format(ks, kt))




if __name__=="__main__":
    main()
