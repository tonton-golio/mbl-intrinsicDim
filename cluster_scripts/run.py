from utils_cluster import *
import argparse
import os
import numpy as np
from numpy import save, load, array

def run(L, seed, output_path):
    ID_and_chi2 = []
    Ws = np.linspace(0.25,7,28)
    for W in Ws:
        H = constructHamiltonian(L = L, W = W, seed=seed)
        _, eigvecs = np.linalg.eigh(H)
        ID, chi2, r1, r2 = nn2(eigvecs)
        ID_and_chi2.append({'ID':ID, 'chi2':chi2, 'r1':r1, 'r2':r2})
    filename = output_path+'2nn_L{}_seed{}'+'.npy'
    np.save(filename, array(ID_and_chi2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run 2NN on Hubbard chain Eigenvectors")
    parser.add_argument('output_path', type=str, help='Output folder of the files')
    parser.add_argument('-L', type=int, default=14, help='The system size (L)')
    parser.add_argument('-S', type=int, help='The seed number')
    args = parser.parse_args()

    # Make sure the output_path directory exists
    try:
        os.makedirs(args.output_path, exist_ok = True)
    except:
        print("Directory '%s' could not be created" % args.output_path)
    
    run(args.L, args.S, args.output_path)
