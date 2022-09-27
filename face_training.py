#import opencv package
from array import array
from enum import unique
from lib2to3.pytree import convert
import cv2

#import Numerical package
import numpy as np
from PIL import Image

#import os to perform Operating System release functionality
import os

from macpath import join, split
#path for face image database
path = 'E:\face_recognision/dataset' # specific path

#os.chdir changes current working directory to a different path
os.chdir("/home/pi/opencv/data/haarcascades")
recognizer = cv2.face.createLBPHFaceRecognizer()
detector = cv2.CascadeClassifier(/home/pi/opencv/data/haarcascades/haarcascades_frontalface_default.xml");

# function to get the image and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = image.open(imagePath).convert('L') #convert it to grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imgPath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSample.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids

print("\n [INFO] Training faces. It will take a few seconds. wait ...")
faces, ids = getImagesAndlabels(path)
recognizer.train(faces, np.array(ids))

#save the model into trainer/trainer.yml
recognizer.SAVE('/HOME/PI/desktop/projectAMB001/code/Face-Recognition/trainer/trainer.yml') 

#print the number of faces trained and end program
print("\n [INFO] {0} faces trained. Existing program".format(len(np.unique(ids))))