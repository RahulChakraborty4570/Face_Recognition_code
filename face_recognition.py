#import opencv package
from tkinter.font import names
import cv2
# import Numerical package
import numpy as np
#Import OS to perform Operating System related functionality
import os

#os.chdir changes current working directory to a different path
os.chdir("/home/pi/opencv/data/haarcascades") 

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('/home/pi/desktop/project/code/Face-Recognition/trainer/trainer.yml')
cascadePath = "/home/pi/opoencv/data/haarcascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#initialise id counter
id = 0

#names related to ids
names = ['None', 'Name 1', 'Name 2', 'Name 3']

# initialize and start realtime video capture 
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) #set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img = cam.read()
    #img = cv2.flip(img, -1) # Flip vertically

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
    )
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence= recognizer.predict(gray[y:y+h, x:x+w])

        # Checking cofidence level;
        if (confidence < 100):
            id = names[id]
            confidence = " {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = " {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
    
    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff # press 'ESC' for exiting video
    if k == 27:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destryAllWindows()