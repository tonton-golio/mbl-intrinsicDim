# Add parent directory to path, so that we can find the utils_cluster module
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils_cluster import *
import argparse
import numpy as np

def run(L, seed, output_path):
    # Fixed high disorder to confirm ID ~> L
    W = 20

    H = constructHamiltonian(L = L, W = W, seed=seed)
    _, eigvecs = np.linalg.eigh(H)

    ID_vs_fraction = {}
    num_averages = 1000
    for fraction in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ID_vs_fraction[fraction] = []
        for a in range(num_averages):
            # Take 'fraction' many eigenstates, random and uniformly
            sample = np.random.choice(range(len(eigvecs)))
            # Compute ID for this sample
            ID, rsquared, nndist = nn2(eigvecs[:,sample])
            ID_vs_fraction[fraction].append([ID,rsquared])
            # We could stop here for fraction = 1.0, but having num_average identical samples
            # makes the analysis script a little easier to read...
            
    filename = output_path+'plateauing_L_{0}_seed_{1}'.format(L, seed)+'.npy'
    np.save(filename, ID_vs_fraction)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Plateauing")
    parser.add_argument('output_path', type=str, help='Output folder of the files')
    parser.add_argument('-L', type=int, default=14, help='The system size (L)')
    parser.add_argument('-S', type=int, help='The seed number')
    args = parser.parse_args()

    # Make sure the output_path directory exists
    os.makedirs(args.output_path, exist_ok = True)
    run(args.L, args.S, args.output_path)
