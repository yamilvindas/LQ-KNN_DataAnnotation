#!/usr/bin/env python3
"""
    Example of feature extraction (first step of our proposed method)
"""
import subprocess

def main():
    with subprocess.Popen(\
                            [
                                'python',\
                                '../src/feature_extraction.py',\
                                '--parameters_file',\
                                '../parameters_files/default_parameters_AE.json'
                            ], stdout=subprocess.PIPE
                         ) as proc:
        for line in proc.stdout:
            line = line.decode("utf-8")
            print(line)

if __name__=='__main__':
    main()
