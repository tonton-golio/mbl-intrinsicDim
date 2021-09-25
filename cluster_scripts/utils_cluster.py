from math import factorial
from numpy import zeros, random, array, sort, argsort, arange, log, vstack, eye
from numpy.linalg import eigh,lstsq
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
	dist_M = array([[sum(abs(a-b)) if index0 < index1 else 0 for index1, b in enumerate(A)] for index0, a in enumerate(A)])
	dist_M += dist_M.T + eye(N)*42
    
    # Calculate mu
	argsorted = sort(dist_M, axis=1)
	mu =  argsorted[:,1]/argsorted[:,0]
	x = log(mu)
    
    # Permutation
	dic = dict(zip(argsort(mu),(arange(1,N+1)/N)))
	y = array([1-dic[i] for i in range(N)])
    
    # Drop bad values (negative y's)
	x,y  = x[y>0], y[y>0]
	y = -1*log(y)
    
    #fit line through origin to get the dimension
	d = lstsq(vstack([x, zeros(len(x))]).T, y, rcond=None)[0][0]

    # Goodness
	chi2, _ = chisquare(f_obs=x*d , f_exp=y, ddof=10)
    
	return d, chi2


