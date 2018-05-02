import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

lefteye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_lefteye_2splits.xml')
righteye_cascade =cv2.CascadeClassifier('haarcascades/haarcascade_righteye_2splits.xml')

smile_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_smile.xml')

cap = cv2.VideoCapture(0)


while True:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.2, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame ,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_frame = frame[y:y+h, x:x+w]

	left_eye = lefteye_cascade.detectMultiScale(roi_gray, 1.8, 5)
	for (lex,ley,leh,lew) in left_eye:
		cv2.rectangle(roi_frame,(lex,ley),(lex+lew,ley+leh),(0,255,0),2)

	right_eye = righteye_cascade.detectMultiScale(roi_gray, 1.8, 5)
	for (rex,rey,reh,rew) in right_eye:
		cv2.rectangle(roi_frame,(rex,rey),(rex+rew,rey+reh),(0,255,255),2)

	smile = smile_cascade.detectMultiScale(roi_gray, 2,5)
	for (sx,sy,sh,sw) in smile:
		cv2.rectangle(roi_frame,(sx,sy),(sx+sh,sy+sw),(0,0,255),2)
	
	cv2.imshow('frame',frame)
	k = cv2.waitKey(1) & 0xff
	if k == 27:
	    break


cap.release()
cv2.destroyAllWindows()