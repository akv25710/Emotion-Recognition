import numpy as np 
from sklearn import svm

X1 = np.loadtxt('extracted_data\\TrainingFeatures.txt')
y1 = np.loadtxt('extracted_data\\TrainingLabels.txt')
X2 = np.loadtxt('extracted_data\\TestingFeatures.txt')
y2 = np.loadtxt('extracted_data\\TestingLabels.txt')


clf = svm.SVC(C=1, kernel = 'linear', probability = True)
clf.fit(X1, y1.flatten())

accuracy = clf.score(X2, y2.flatten())

print accuracy 




