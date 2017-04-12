import numpy as np 
import cv2
import dlib
import glob
import math
import random
import os


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def get_landmarks(img):
	detections = detector(img, 1)
	
	for k,d in enumerate(detections):
		shape = predictor(img, d)
		xlist = []
		ylist = []

		for i in range (1,68):
			xlist.append(float(shape.part(i).x)) 
			ylist.append(float(shape.part(i).y))

		xmean = np.mean(xlist)
		ymean = np.mean(ylist)

		
		if xlist[26] == xlist[29]:   #xlist[26] -> top of the nose bridge & xlist[29] -> tip of the nose
			anglenose = 0

		else:
			anglenose = int( math.atan( (ylist[26]-ylist[29]) / (xlist[26]-xlist[29]) )*180/math.pi )

		if anglenose < 0:
			anglenose += 90
		else:
			anglenose -= 90

		
		features = []
		for i in xrange(len(xlist)):
			if xlist[i] == xmean:
				angle = 90 - anglenose
			else:
				angle = int( math.atan( (ylist[i]-ymean)/(xlist[i]-xmean) )*(180/math.pi) ) - anglenose

			features.append(angle)
	return features

def build_data():
	training_data = []
	training_labels = []
	testing_data = []
	testing_labels = []

	emotions = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]

	for emotion in emotions:
		files = glob.glob('dataset\\%s\\*' %emotion)
		np.random.shuffle(files)

		training = files[:int(len(files)*0.8)] 
		prediction = files[-int(len(files)*0.2):] 

		for item in training:
			img = cv2.imread(item)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			clahe_img = clahe.apply(gray)
			features = get_landmarks(clahe_img)
			training_data.append((features))
			training_labels.append(emotions.index(emotion))

		for item in prediction:
			img = cv2.imread(item)
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			clahe_img = clahe.apply(gray)
			features = get_landmarks(clahe_img)
			testing_data.append(features)
			testing_labels.append(emotions.index(emotion))	


	return training_data, training_labels, testing_data, testing_labels

training_data, training_labels, testing_data, testing_labels = build_data()


X1 = open('extracted_data\\TrainingFeatures.txt','w')
X2 = open('extracted_data\\TestingFeatures.txt','w')
Y1 = open('extracted_data\\TrainingLabels.txt','w')
Y2 = open('extracted_data\\TestingLabels.txt','w')

print len(training_data)
print len(testing_data)

 
for i in xrange(len(training_data)):
	for j in xrange(67):
		X1.write(str(training_data[i][j])+" ")
	X1.write("\n")	
for i in xrange(len(testing_data)):
	for j in xrange(67):
		X2.write(str(testing_data[i][j])+" ")
	X2.write("\n")	
for i in xrange(len(training_labels)):
	Y1.write(str(training_labels[i]))
	Y1.write("\n")	
for i in xrange(len(testing_labels)):
	Y2.write(str(testing_labels[i]))
	Y2.write("\n")	

X1.close()
X2.close()
Y1.close()
Y2.close()



