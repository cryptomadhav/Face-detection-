# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:25:21 2019

@author: Madhav
"""

# Smile Recognition

# Importing the libraries
import cv2

# Loading the cascades
# Complete path to file is required
face_cascade = cv2.CascadeClassifier("C:\\Users\\Madhav\\Documents\\computer vision\\Module_1_Face_Recognition\\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("C:\\Users\\Madhav\\Documents\\computer vision\\Module_1_Face_Recognition\\haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("C:\\Users\\Madhav\\Documents\\computer vision\\Module_1_Face_Recognition\\haarcascade_smile.xml")

#Defining function that will do the detections of face
def detect(gray, frame):#gray of frame , original frame to draw on
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)#image, scale factor to reduce image, mim number of neighbors
    for (x, y, w, h) in faces:
        #draw rectangles on faces
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y: y + h, x : x + w]
        roi_color = frame[y: y + h, x : x + w]
        #detect eyes in roi_gray
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)#detect eyes using eyes classifier
        for (ex, ey, ew, eh) in eyes:
            #draw rectangles on faces
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)
        #Detect smile in roi_gray
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.2, 50)#detect eyes using eyes classifier
        for (sx, sy, sw, sh) in smiles:
            #draw rectangles on faces
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 1)
        
    return frame

#Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0) # 0 - internal webcam, 1 - external webcam

#Apply detect on each frame of video_capture
while True:
    _, frame = video_capture.read() #only need second value i.e last frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    # Display modified images in window
    cv2.imshow("Video", canvas)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
video_capture.release()
cv2.destroyAllWindows()
