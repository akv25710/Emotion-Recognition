import numpy as np 
from scipy.special import expit
from scipy import optimize
import pandas as pd 

X1 = np.loadtxt('extracted_data\\TrainingFeatures.txt',dtype=float)
y1 = np.loadtxt('extracted_data\\TrainingLabels.txt')
X2 = np.loadtxt('extracted_data\\TestingFeatures.txt',dtype=float)
y2 = np.loadtxt('extracted_data\\TestingLabels.txt')
y1 = np.array([y1]).T
y2 = np.array([y2]).T

 


def h(mytheta, myX):
	return expit(np.dot(myX, mytheta))

def ComputeCost(mytheta, myX, myy, alpha=0):
	m = X1.shape[0]
	term1 = np.dot(myy.T, np.log(h(mytheta, myX)))
	term2 = np.dot(1-(myy).T, np.log(1-h(mytheta, myX)))
	reg_term = float(alpha)/2 * np.sum(np.dot(mytheta[1:].T, mytheta[1:]))
	return float(-1/m)*(term2+term1+reg_term)



def optimizeTheta(mytheta, myX, myy, alpha=0):
	 i=0
	 while i < 8:
	 	theta = np.zeros((myX.shape[1],1))
	 	tempy = np.zeros( (myy.shape[0],myy.shape[1]))
	 	for j in xrange(tempy.shape[0]):
	 		if y1[j]==i:
	 			tempy[j]=1

		res = optimize.fmin(ComputeCost,x0=theta,args=(myX,tempy,alpha),maxiter=100000,full_output=True)
		ans = res[0]
		#print ans

		for k in xrange(ans.shape[0]):
			mytheta[k,i] = ans[k]

		i+=1

learned_thetas = np.zeros((X1.shape[1],8))
optimizeTheta(learned_thetas,X1,y1,0.05)
#print learned_thetas


def Accuracy(learned_thetas,myX,myy):
	m =myX.shape[0]
	correct = 0
	for i in xrange(m):
		val = -1
		res = -1
		for j in xrange(7):
			myTheta = learned_thetas[:,j]
			#print myTheta
			myTheta = np.array([myTheta]).T
			tempX = np.array([myX[i]])
			#print tempX
			ans = h(myTheta,tempX)
			#print ans
			if ans > res :
				res = ans
				val = j

		if val == myy[i]:
			correct += 1

        print "Correctly Classified", correct
        print "Total", m
        print "Accuracy", float(correct)/float(m) * 100


Accuracy(learned_thetas,X2,y2)

	

