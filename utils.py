import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from tqdm import tqdm
import fssa
from scipy.stats import chisquare
    
## Building the Hamiltonian 
def binaryConvert(x=5, L=4):
	'''
	Convert base-10 integer to binary and adds zeros to match length
	_______________
	Paramters:
		x : base-10 integer
		L : length of bitstring 
	_______________
	returns: 
		b : Bitstring 
	'''
	b = bin(x).split('b')[1]
	while len(b) < L:
		b = '0'+b
	return b

def nOnes(bitstring='110101'):
	'''
	Takes binary bitstring and counts number of ones
	_______________
	Parameters:
	bitstring : string of ones and zeros
	_______________
	returns: 
		counta : number of ones
	'''
	counta = 0
	for i in bitstring:
		if i=='1':
			counta += 1
	return counta

def binomial(n=6, pick='half'):
	'''
	find binomial coefficient of n pick k,
	_______________
	Parameters:
		n : total set
		pick : subset 
	_______________
	returns:
		interger
	'''
	if pick == 'half':
		pick = n//2

	return int(factorial(n) / factorial(n-pick) / factorial(pick))

def basisStates(L=5):
	'''
	Look for basis states
	_______________
	Parameters:
		L : size of system; integer divible by 2
	_______________
	Returns:
		dictionaries (State_to_index, index_to_State)
	'''
	if L%2!=0:
		print('Please input even int for L')

	s2i = {} # State_to_index
	i2s = {} # index_to_State

	index = 0
	for i in range(int(2**L)): # We could insert a minimum
		binary = binaryConvert(i, L)
		ones = nOnes(binary)

		if ones == L//2:
			s2i[binary] = index
			i2s[i] = binary
			index +=1

	return (s2i, i2s)

def energyDiagonal(bitString='010', V=[0,0.5,1] , U=1.0):
	'''
	Diagonal of Hamiltonian with periodic boundary conditions 
	______________
	Parameters:
		bitString : ones and zeros; string 
		V : onsite potentials for each site; list of floats
		U : interaction; float
	______________
	returns :
		E : diagonal of H; list of floats
	'''
	E = 0
	for index, i in enumerate(bitString):
		if i =='1':
			E += V[index]
			try:
				if bitString[index+1] == '1':
					E += U
			except IndexError:
				if bitString[0] == '1':
					E += U
	return E


def construct_potential(L = 4, W = 2, seed=42, disorder_distribution ='uniform'):
	np.random.seed(seed)
	if disorder_distribution == 'uniform':
		V = np.random.uniform(-1,1,size=L) * W
	elif disorder_distribution == 'bimodal':
		V = np.concatenate([np.random.normal(W, size=L), np.random.normal(-W, size=L)])
		np.random.shuffle(V)
	elif disorder_distribution == 'normal':
		V = (np.random.normal(0,1,size=L)) * W
	elif disorder_distribution =='sinusoidal':
		V = np.cos(np.arange(L)*np.pi)
	else:
		V = np.random.uniform(-1,1,size=L) * W
	return V


def constructHamiltonian(L = 4, W = 2, U = 1.0, t = 1.0, seed=42, periodic_boundary_conditon=True, disorder_distribution ='uniform'):
	'''
	Constructs the Hamiltonian matrix
	________________
	Parameters:
		L : size of system; integer divible by 2
		W : disorder strength; float
		U : Interaction; flaot
		t : hopping term; float
		seed : seed for random
	________________
	returns:
		Hamiltonian
	'''
	V = construct_potential(L = L, W = W, seed=seed, disorder_distribution =disorder_distribution)
	num_states = binomial(L)
	H = np.zeros((num_states,num_states))
	(s2i, i2s) = basisStates(L)
    
	for key in s2i.keys():
		H[s2i[key],s2i[key]] = energyDiagonal(key, V, U)  # fill in the diagonal with hop hopping terms
		for site in range(L):
			try:
				if (key[site] == '1' and key[site+1]== '0'):
					new_state = key[:site] + '0' + '1' + key[site+2:]
					H[s2i[new_state], s2i[key]], H[s2i[key], s2i[new_state]] = t ,t
			except IndexError: # periodic boundary conditions
				if periodic_boundary_conditon == True:
					if (key[site] == '1' and key[0]== '0'):
						new_state = '1' + key[1:site] + '0'
						H[s2i[new_state], s2i[key]], H[s2i[key], s2i[new_state]] = t ,t
				else:
					pass
	return H

def buildDiagSave(L = 10, num_seeds = 10, ws = [1,2,3], location = 'data/raw/'):
    '''
    builds Hamiltonians, diagonalizes and saves eigvecs    
    '''
    for W in tqdm(ws):
        for seed in range(num_seeds):
            filename = location+'eigvecs-L-{}-W-{}-seed-{}.npy'.format(L, round(W,2), seed)
            try:
                np.load(filename)
                print(filename,'exists')
            except:
                H = constructHamiltonian(L = L, W = W, seed=seed)
                _, eigvecs = np.linalg.eigh(H)
                np.save(filename, eigvecs)
    return 'success'


## EigenComponent dominance
def count_lower_than(lst, lim):
    counta = 0
    for i in lst:
        if i < lim:
            counta +=1
    return counta

def eigenC_analysis(ws, num_lims=8,L=8, num_seeds=10, location='data/raw/'):
    lims=np.logspace((1-num_lims),0,num_lims)
    data_dict = {}
    
    #maxs, lower_than = np.zeros((steps, seeds)), np.zeros((steps, num_lims, seeds))
    for index0, W in enumerate(ws):
        data_dict[W] = {}
        for index1, seed in enumerate(range(num_seeds)):
            data_dict[W][seed] = {'lim':{}}
            
            filename = location+'eigvecs-L-{}-W-{}-seed-{}.npy'.format(L, W, seed)
            eigs = abs(np.load(filename).flatten())
            data_dict[W][seed]['max'] = np.max(eigs)
            
            for index2, lim in enumerate(lims):
                data_dict[W][seed]['lim'][lim] = count = count_lower_than(eigs, lim)
    
    maxs = np.array([[data_dict[W][seed]['max'] for W in ws] for seed in range(num_seeds)])
    below_lims = np.array([[[data_dict[W][seed]['lim'][lim] for seed in range(num_seeds)] for lim in lims]for W in ws])
    return maxs, below_lims/binomial(L)**2, lims


# 2NN
def nn2(A, return_R1=False, return_xy=False):
	'''
    Find intrinsic dimension (ID) via 2-nearest-neighbours

    https://www.nature.com/articles/s41598-017-11873-y
    https://arxiv.org/pdf/2006.12953.pdf
    _______________
    Parameters:
        eigvecs
        plot : create a plot; boolean; dafault=False
    _______________
    Returns:
        d : Slope
        quality: chiSquared fit-quality
    
    '''
	
	N  = len(A)
    #Make distance matrix
	dist_M = np.array([[sum(abs(a-b)) if index0 < index1 else 0 for index1, b in enumerate(A)] for index0, a in enumerate(A)])
	dist_M += dist_M.T + np.eye(N)*42
    
    # Calculate mu
	argsorted = np.sort(dist_M, axis=1)
	mu =  argsorted[:,1]/argsorted[:,0]
	R1 = argsorted[:,0]
	x = np.log(mu)
    
    # Permutation
	dic = dict(zip(np.argsort(mu),(np.arange(1,N+1)/N)))
	y = np.array([1-dic[i] for i in range(N)])
    
    # Drop bad values (negative y's)
	x,y  = x[y>0], y[y>0]
	y = -1*np.log(y)
    
    #fit line through origin to get the dimension
	d = np.linalg.lstsq(np.vstack([x, np.zeros(len(x))]).T, y, rcond=None)[0][0]

    # Goodness
	chi2, _ = chisquare(f_obs=x*d , f_exp=y, ddof=10)

	if return_R1==True:
		return d, chi2, R1

	elif return_xy == True:
		return d, chi2, x,y

	else:
		return d, chi2

def nn2_loop(ws, num_seeds, Ls, location='data/raw/'):
	'''run 2nn for many disorders and many seeds...returns ID and chi2 in dict'''
	ID_and_chi2 = {}
	for L in Ls:
		ID_and_chi2[L]={}
		for W in tqdm(ws):
			ID_and_chi2[L][W] = {'ID' :{}, 'chi2' :{} }
			for seed in range(num_seeds):
				filename = location+'eigvecs-L-{}-W-{}-seed-{}.npy'.format(L, W, seed)
				eigs = np.load(filename)
				d, chi2 = nn2(eigs, plot=False)
				ID_and_chi2[L][W]['ID'][seed] = d
				ID_and_chi2[L][W]['chi2'][seed] = chi2
	return ID_and_chi2

def scale_collapse2(data, ws, Ls=[6,8],rho_c0=3.5,
	nu0=2., zeta0=2., skip_initial = 2, drop_ls = 0):
    data=data[:,skip_initial:]
    ws=ws[skip_initial:]
    da = data / 100
    res = fssa.autoscale(l=Ls, rho=ws, a=data, da=da, rho_c0=rho_c0, nu0=nu0, zeta0=zeta0)
    print('autoscale done')
    fig, ax = plt.subplots()
    for index, L in enumerate(Ls):
        ax.plot(ws, data[index], label='L={}'.format(L))
    ax.legend()
    ax.set_xlabel('Disorder strength, $W$', fontsize=14)
    ax.set_ylabel('$\overline{\mathcal{D}_{int}}$', fontsize=14)
    axin = ax.inset_axes([0.5, 0.5, 0.45, 0.45])
    scaled = fssa.scaledata(l=Ls, rho=ws, a=data, da=da, rho_c=res['rho'], nu=res['nu'], zeta=res['zeta'])
    print('Scale data done')
    X = scaled[0]
    Y = scaled[1]
    for index, L in enumerate(Ls):
        axin.plot(X[index], Y[index])

    quality = fssa.quality(X,Y,da)
    fig.suptitle('$\overline{\mathcal{D}_{int}}$ with collapse on inset',fontsize=16)
    return res


def random_index(x=5, N=252):
    a = (np.random.random_sample(x)*N).astype(int)
    while x!=len(set(a)):
        a = np.append(a,int(np.random.random_sample()*N))
    return a  # maybe use np.random.sample()

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, np.sqrt(variance)

def plateau(Ls = [6,8,10], W = 10, seed= 42,runs_lst = [2000,1000,200]):
	data_dict = {}
	data_dict['params'] = dict(zip(Ls, runs_lst))
	data_dict['params']['W'] = W
	data_dict['params']['seed'] = seed
		
	for L, runs in zip(Ls, runs_lst):
		data_dict[L] = {}
		N = binomial(L)
		spacing = np.linspace(0.15,.9,12)
		num_samples_lst = (spacing*N).astype(int)
		inner_data_dict = {}

		vals, vecs = np.linalg.eigh(constructHamiltonian(L=L, W=W, seed=seed, periodic_boundary_conditon=True))
		data_dict[L][1.0] = {'id':nn2(vecs)[0], 'std':0}

		for num_samples in tqdm(num_samples_lst):
			tmp_id, tmp_q = [], []
			for run in range(runs):
				sample_index = random_index(num_samples, N)
				vecs_sample = vecs[:,sample_index]
				d, q = nn2(vecs_sample)
				tmp_id.append(d)
				tmp_q.append(q)

			mean_id, std = weighted_avg_and_std(tmp_id, tmp_q)
			data_dict[L][round(num_samples/N,3)] = {'id':mean_id, 'std':std}
	return data_dict