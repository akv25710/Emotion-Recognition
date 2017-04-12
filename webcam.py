import cv2 
import dlib

video_capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_dectector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
	ret, frame = video_capture.read()
	grav = cvt.cvtColor('frame', cv2.COLOR_BGR2GRAY)
	clahe = cv2,createCLAHE(clipLimit = 2.0, title)