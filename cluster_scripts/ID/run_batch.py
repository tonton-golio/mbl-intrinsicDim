# Add parent directory to path, so that we can find the utils_cluster module
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils_cluster import *
import argparse
import numpy as np

def run(L, startseed, endseed, output_path):
    Ws = np.concatenate([np.arange(1,2.6,0.2), np.arange(2.6,4.55,0.05), np.arange(4.7,6.2,0.2)])

    for seed in range(startseed, endseed):
        data = []
        evals = []
        nn_indices = []
        for w,W in enumerate(Ws):
            this_seed = int(L * 1e7 + w * 1e5 + seed)

            H = constructHamiltonian(L = L, W = W, seed=this_seed)
            eigvals, eigvecs = np.linalg.eigh(H)
            ID, rsquared, nndist, nn_indxs = nn2(eigvecs)
            data.append({'ID':ID, 'rsquared':rsquared, 'nndist':nndist})
            evals.append(eigvals)
            nn_indices.append(nn_indxs)
    
        filename = output_path+'2nn_L_{0}_seed_{1}.npy'.format(L, seed)
        np.save(filename, np.array(data))
        filename = output_path+'spectrum_L_{0}_seed_{1}.npy'.format(L, seed)
        np.save(filename, np.array(evals))
        filename = output_path+'nn_indices_L_{0}_seed_{1}.npy'.format(L, seed)
        np.save(filename, np.array(nn_indices))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run 2NN on Hubbard chain Eigenvectors")
    parser.add_argument('output_path', type=str, help='Output folder of the files')
    parser.add_argument('-L', type=int, default=14, help='The system size (L)')
    parser.add_argument('-Ss', type=int, help='The start seed number')
    parser.add_argument('-Se', type=int, help='The end seed number')
    args = parser.parse_args()

    # Make sure the output_path directory exists
    os.makedirs(args.output_path, exist_ok = True)
    run(args.L, args.Ss, args.Se, args.output_path)
