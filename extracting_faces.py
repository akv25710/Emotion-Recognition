import cv2
import glob
import numpy as np

faceDet  = cv2.CascadeClassifier("haarcascade\\haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("haarcascade\\haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("haarcascade\\haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("haarcascade\\haarcascade_frontalface_alt_tree.xml")

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

def crop_faces(emotion):
	files = glob.glob("sorted_set\\%s\\*" %emotion)
	filenum = 0

	for file in files:
		frame = cv2.imread(file)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		face  = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors = 10, minSize=(5,5), flags = cv2.CASCADE_SCALE_IMAGE)
		face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors = 10, minSize=(5,5), flags = cv2.CASCADE_SCALE_IMAGE)
		face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors = 10, minSize=(5,5), flags = cv2.CASCADE_SCALE_IMAGE)
		face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors = 10, minSize=(5,5), flags = cv2.CASCADE_SCALE_IMAGE)


		if len(face) == 1:
			facefeatures = face
		elif len(face2) == 1:
			facefeatures = face2	
		elif len(face3) == 1:
			facefeatures = face3
		elif len(face4) == 1:
			facefeatures = face4
		else:
			facefeatures = ""				

		for (x,y,w,h) in facefeatures:
			gray = gray[y:y+h, x:x+w]

			try:
				img = cv2.resize(gray, (350,350))
				cv2.imwrite("dataset\\%s\\%s.jpg" %(emotion, filenum), img)
			except:
				pass

		filenum += 1		

for emotion in emotions:
	crop_faces(emotion)