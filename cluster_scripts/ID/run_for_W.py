# Add parent directory to path, so that we can find the utils_cluster module
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils_cluster import *
import argparse
import numpy as np

def run(L, seed, W, output_path):
    data = []
    this_seed = int(L * 1e7 + w * 1e5 + seed)

    H = constructHamiltonian(L = L, W = W, seed=seed)
    eigvals, eigvecs = np.linalg.eigh(H)
    ID, rsquared, nndist = nn2(eigvecs)
    data.append({'ID':ID, 'rsquared':rsquared, 'nndist':nndist})
    
    filename = output_path+'2nn_L_{0}_seed_{1}_W_{2}.npy'.format(L, seed, W)
    np.save(filename, np.array(data))
    filename = output_path+'spectrum_L_{0}_seed_{1}_W_{2}.npy'.format(L, seed,W)
    np.save(filename, np.array(eigvals))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run 2NN on Hubbard chain Eigenvectors")
    parser.add_argument('output_path', type=str, help='Output folder of the files')
    parser.add_argument('-L', type=int, help='The system size (L)')
    parser.add_argument('-S', type=int, help='The seed number')
    parser.add_argument('-W', type=float, help='The disorder strength') 
    args = parser.parse_args()

    # Make sure the output_path directory exists
    os.makedirs(args.output_path, exist_ok = True)
    run(args.L, args.S, args.W, args.output_path)
