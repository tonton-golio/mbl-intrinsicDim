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

def constructHamiltonian(L = 4, W = 2, U = 1.0, t = 1.0, seed=42):
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
	np.random.seed(seed)
	V = np.random.uniform(-1,1,size=L) * W
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
				if (key[site] == '1' and key[0]== '0'):
					new_state = '1' + key[1:site] + '0'
					H[s2i[new_state], s2i[key]], H[s2i[key], s2i[new_state]] = t ,t
	return H

def buildDiagSave(L = 10, num_seeds = 10, ws = [1,2,3], location = 'data/raw/'):
    '''
    builds Hamiltonians, diagonalizes and saves eigvecs    
    '''
    for W in tqdm(ws):
        for seed in range(num_seeds):
            H = constructHamiltonian(L = L, W = W, seed=seed)
            _, eigvecs = np.linalg.eigh(H)
            filename = location+'eigvecs-L-{}-W-{}-seed-{}.npy'.format(L, round(W,2), seed)
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

def eigenC_plots(below_lims, maxs, ws,
                 lims = np.logspace((1-8),0,8),
                 num_seeds =10, L=8,
                 colors = 'orange, lightblue, salmon, yellowgreen, grey, purple'.split(', ')
                ):
    # Plot 1: proportion below threshhold
    mean_below = np.mean(below_lims,axis=2).T[::-1]
    fig, ax  = plt.subplots(2,1, sharex=True, 
                           gridspec_kw={'height_ratios':[2,1]},
                           figsize=(10,6))
    
    for i, color in zip(range(len(mean_below)), colors):
        ax[0].fill_between(ws, mean_below[i], mean_below[i+1],
                         label=lims[::-1][i],
                         color=color, alpha=.3)
    
    # Plot 2: Maxs
    for index, i in enumerate(maxs):
        ax[1].scatter([ws[index]]*num_seeds, 1-i, c='b', alpha=2/num_seeds)
        ax[1].scatter([ws[index]], 1-np.mean(i), c='r', alpha=0.9)

    # Labels and such
    ax[0].legend(bbox_to_anchor=(1, 1.), fontsize=12)
    ax[0].set_ylabel('Proportion of $|\lambda_c|<\zeta$ ', fontsize=14)
    ax[1].legend(["point", "mean"],#bbox_to_anchor=(0.2, .25),
    	facecolor='white', framealpha=1,
        fontsize=14)
    plt.xlabel('Disorder strength, $W$', fontsize=14)
    ax[1].set_ylabel('$1-max(|\lambda_c|)$', fontsize=14)    
    plt.suptitle('Eigencomponent, $\kappa$, dominance', fontsize=17)

# 2NN
def nn2(A, plot=False):
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

	if plot==True:
		plt.scatter(x,y, c='purple', alpha=0.5)
		plt.plot(x,x*d, c='orange', ls='-')
		plt.title('2NN output for single realization', fontsize=0)
		plt.xlabel('$\log(\mu)$', fontsize=14)
		plt.ylabel('$1 - F(\mu)$', fontsize=14)
		plt.grid()
        #plt.savefig('figures/2nnSingle_L{}'.format(10),dpi=400,bbox_inches='tight')
    
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