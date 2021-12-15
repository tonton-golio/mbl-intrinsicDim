# Add parent directory to path, so that we can find the utils_cluster module
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils_cluster import *
import argparse
import numpy as np

def collect(L, seed, output_path):
    alldata = []
    Ws = np.concatenate([np.arange(1,2.6,0.2), np.arange(2.6,4.55,0.05), np.arange(4.7,6.2,0.2)])
    for W in Ws:
        filename = output_path+'2nn_L_%d_seed_%d_W_%.1f.npy'%(L, seed, W)
        data = np.atleast_2d(np.load(filename, allow_pickle=True))[0][0]
        alldata.append(data)

    filename = output_path+'2nn_L_{0}/2nn_L_{0}_seed_{1}.npy'.format(L, seed)
    np.save(filename, np.array(alldata))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run 2NN on Hubbard chain Eigenvectors")
    parser.add_argument('output_path', type=str, help='Output folder of the files')
    parser.add_argument('-L', type=int, help='The system size (L)')
    parser.add_argument('-S', type=int, help='The seed number')
    args = parser.parse_args()

    # Make sure the output_path directory exists
    os.makedirs(args.output_path, exist_ok = True)
    collect(args.L, args.S, args.output_path)
