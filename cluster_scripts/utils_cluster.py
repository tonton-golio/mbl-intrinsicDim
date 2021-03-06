from math import factorial
from numpy import zeros, random, array, sort, argsort, arange, log, eye, sum, mean
import numpy as np
from scipy.optimize import curve_fit

    
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
	random.seed(seed)
	V = random.uniform(-1,1,size=L) * W
	num_states = binomial(L)
	H = zeros((num_states,num_states))
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

## 2NN
def linear_origin_bound(x, a):
    return a * x

def nn2(A, distance_metric='L1'):
	'''
    Find intrinsic dimension (ID) via 2-nearest-neighbours
    https://www.nature.com/articles/s41598-017-11873-y
    https://arxiv.org/pdf/2006.12953.pdf
    _______________
    Parameters:
        eigvecs
    _______________
    Returns:
        d : Slope
    '''
	N  = len(A)
    #Make distance matrix
	if distance_metric == 'L1':
		dist_M = array([[sum(abs(a-b)) if index0 < index1 else 0 for index1, b in enumerate(A)] for index0, a in enumerate(A)])
	else:
		dist_M = array([[max(abs(a-b)) if index0 < index1 else 0 for index1, b in enumerate(A)] for index0, a in enumerate(A)])
	# Add an offset to the diagonal to avoid having a nearest neighbor distance of 0
	dist_M += dist_M.T + eye(N)*1e6
    
    # Calculate mu
	M_sorted_indices = np.argsort(dist_M, axis=1)
	r1_index = M_sorted_indices[:,0]
	Msorted = np.take_along_axis(dist_M, M_sorted_indices, axis=1)
	r1, r2 = Msorted[:,0], Msorted[:,1]
	mu = r2/r1
	x = log(mu)
    
    # Permutation
	dic = dict(zip(argsort(mu),(arange(1,N+1)/N)))
	y = array([1-dic[i] for i in range(N)])
    
    # Drop bad values (negative y's)
	x,y  = np.nan_to_num(x[y>0]), np.nan_to_num(y[y>0])
	y = -1*log(y)
    
    #fit line through origin to get the dimension
	#fit = lstsq(vstack([x, zeros(len(x))]).T, y, rcond=None)
	popt, pcov = curve_fit(linear_origin_bound, x, y)
	d = popt[0]

    # Goodness of fit with R^2
	rsquared = 1 - sum((y-popt[0]*x)**2)/ sum((y-mean(y))**2)

	return d, rsquared, r1, r1_index
