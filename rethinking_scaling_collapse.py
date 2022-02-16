import numpy as np
from scipy.optimize import minimize

def distance(a,b, metric='L2'):
	metric_dict = {
		'L1' : lambda a,b: abs(a-b),
		'L2' : lambda a,b:  (a-b)**2
	}

	return metric_dict[metric](a,b)


def cost_function(data, metric = 'L1'):
	C = np.zeros(data.shape()[1])
	for index in range(data.shape()[1]):
		# data[index] will look like: [6,7,4,5,6]
		mean = np.mean(data[index])
		point_cost = sum([distance(i,mean) for i in data[index]])
		C[index] = point_cost
	return sum(C)


def scaling_func1(arr, a, b):
	return a*arr**(-b)

def scaling_func2(arr, c, d):
	pass

def scaling_func3(arr, e, f):
	pass

def scale_x(x, param1, param2):
	return scaling_func1(x, param1, param2)

def scale_Y(Y, param3, param4):
	new_Y = np.zeros(Y.shape())
	for index, row in enumerate(Y):
		new_Y[index] = scaling_func2(row, param3, param4)
	return new_Y




def scale_and_evaluate(x, Y, 
	param1, param2, param3, param4):
	x_scaled = scale_x(x, param1, param2)
	Y_scaled = scale_Y(y, param3, param4)
	cost = cost_function(Y, metric="L1")

	return cost


def auto_scale(x,Y, p0):
	popt, pcov = minimize(scale_and_evaluate, x, Y, p0=p0)

	return popt





















