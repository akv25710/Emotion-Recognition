import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.optimize
import scipy.misc
import random
import matplotlib.cm as cm
from scipy.special import expit
import itertools

X1 = np.loadtxt('extracted_data\\TrainingFeatures.txt',dtype=float)
y1 = np.loadtxt('extracted_data\\TrainingLabels.txt', dtype=int)
X2 = np.loadtxt('extracted_data\\TestingFeatures.txt',dtype=float)
y2 = np.loadtxt('extracted_data\\TestingLabels.txt', dtype=int)

y1 = np.array([y1]).T
y2 = np.array([y2]).T

X1 = np.insert(X1,0,1,axis=1)
X2= np.insert(X2,0,1,axis=1)

input_layer_size = 67
hidden_layer_size = 50
output_layer_size = 8 
n_training_samples = X1.shape[0]
m = n_training_samples
print m



def flattenParams(thetas_list):
    flattened_list = [ mytheta.flatten() for mytheta in thetas_list ]
    #print flattened_list
    combined = list(itertools.chain.from_iterable(flattened_list))
    #print combined
    assert len(combined) == (input_layer_size+1)*hidden_layer_size + \
                            (hidden_layer_size+1)*output_layer_size
    return np.array(combined).reshape((len(combined),1))

def reshapeParams(flattened_array):
    theta1 = flattened_array[:(input_layer_size+1)*hidden_layer_size] \
            .reshape((hidden_layer_size,input_layer_size+1))
    theta2 = flattened_array[(input_layer_size+1)*hidden_layer_size:] \
            .reshape((output_layer_size,hidden_layer_size+1))
    
    return [ theta1, theta2 ]

def flattenX(myX):
    return np.array(myX.flatten()).reshape((n_training_samples*(input_layer_size+1),1))

def reshapeX(flattenedX):
    return np.array(flattenedX).reshape((n_training_samples,input_layer_size+1))
 

def PropagateForward(row, theta):
	features = row
	zs_as_per_layer = []
	for i in xrange((len(theta))):
		Theta = theta[i]
		z = np.dot(Theta, features.T).reshape((Theta.shape[0],1))
		a = expit(z)
		zs_as_per_layer.append((z,a))
		if i == len(theta)- 1:
			return np.array(zs_as_per_layer)	
		a = np.insert(a,0,1)
		features = a.T


def ComputeCost(theta , X, y, alpha=0):
	theta = reshapeParams(theta)
	X = reshapeX(X)
	total_cost = 0
	for i in xrange(m):
		row = X[i,:]
		hs = PropagateForward(row, theta)[-1][1]
		tmpy = np.zeros((output_layer_size,1))
		tmpy[y[i]] = 1
		term1 = -np.dot(tmpy.T, np.log(hs))
		term2 = -np.dot((1-tmpy).T, np.log(1-hs))
		cost = term1+term2
		total_cost += cost

	reg_cost = 0
	for Theta in theta:
		reg_cost += np.sum(Theta*Theta)

	reg_cost *= float(alpha)/(2*m)

	return float((1./m)*total_cost) + reg_cost


def sigmoidGradient(z):
	temp = expit(z)
	return temp*(1-temp)

def RandomInitialisation(eps):
	th1 = np.random.uniform(-eps,eps,(hidden_layer_size,input_layer_size+1))
	th2 = np.random.uniform(-eps,eps,(output_layer_size,hidden_layer_size+1))
	th = [th1,th2]
	return th


def backPropagate(theta, X, y, alpha=0):
	theta = reshapeParams(theta)
	X = reshapeX(X)
	theta1 = theta[0]
	theta2 = theta[1]
	Delta1 = np.zeros((hidden_layer_size,input_layer_size+1))
	Delta2 = np.zeros((output_layer_size,hidden_layer_size+1))
	for irow in xrange(m):
		#print irow 
		myrow = X[irow]
		a1 = np.array([myrow])
		#print a1
		temp = PropagateForward(myrow, theta)
		z2 = temp[0][0]
		a2 = temp[0][1]
		#print a2
		z3 = temp[1][0]
		a3 = temp[1][1]
		#print a3
		tmpy = np.zeros((output_layer_size,1))
		tmpy[y[irow]] = 1
		delta3 =  a3 - tmpy
		#print delta3
		g = sigmoidGradient(z2)
		g = np.insert(g,0,1)
		g = np.array([g]).T
		#print g 
		delta2 = np.dot(theta2.T, delta3) * g
		delta2 = delta2[1:]
		#print delta2
		a2 = np.insert(a2,0,1,axis=0)
		Delta2 += np.dot(delta3, a2.T)
		Delta1 += np.dot(delta2, a1)
		

		

	D1 = Delta1/float(m)
	D2 = Delta2/float(m)

	D1[:,1:] = D1[:,1:] + (float(alpha)/m)*theta[0][:,1:]
	D2[:,1:] = D2[:,1:] + (float(alpha)/m)*theta[1][:,1:]	

	return flattenParams([D1,D2]).flatten()


def neuralNetwork(X,y,alpha=0):
	thetas = flattenParams(RandomInitialisation(0.5))
	X = flattenX(X)
	result = scipy.optimize.fmin_cg(ComputeCost, x0=thetas, fprime=backPropagate, args=(X,y,alpha), maxiter=5000, disp=True, full_output=True)
	return reshapeParams(result[0])

learned_thetas = neuralNetwork(X1,y1,0.05)
print learned_thetas

def accuracy(X,y):
	total = X.shape[0]
	correct = 0
	for i in xrange(total):
		ans = PropagateForward(X[i],learned_thetas)[-1][1]
		res = 0
		val = -1
		for j in xrange(len(ans)):
			if ans[j] > res:
				res = ans[j]
				val = j

		if val==y[i]:
			correct+=1

	print "Total : ", total
	print "Correct : ", correct
	print "Accuracy : ", float(correct)/float(total) * 100



accuracy(X2,y2)