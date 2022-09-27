#import opencv package
import cv2

import os

#define video capture object with cam variable
cam = cv2.VideoCapture(0)
cam.set(3, 640) #set video width
cam.set(4, 480) #set video height

#define cascadeClassifier function with haarcascade_frontalface_default.xml file
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#For each person's face, enter id no
face_id = input('\n Enter user ID and press Enter key ')
print("\n [INFO] Initializing face capture process. Face the camera and wait ...")

# Initialize individual sampling face count
count = 0
while (True) :
    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    #draw rectangle around detected faces with blue color
    for (x,y,w,h) in faces:
        cv2.rectangular(img, (x,y), (x+w,y+h), (255,0,), 2)
        count += 1
        # save the captured image in datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + ' . ' + str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # press ESC for existing video
    if k == 27:
        break
    elif count >= 30: # Take30 face sample and stop video
        break

print("\n [INFO] Existing program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()