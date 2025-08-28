import cv2
from cvzone.HandTrackingModule import HandDetector
import time
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 101
folder = 'Data/U'

while True:
    success,img = cap.read()
    hands,img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255
        imgcrop = img[y-offset:y+h+offset,x-offset:x+w+offset]

        try:
            aspectRatio = h/w
            if aspectRatio>1:
                k = imgSize/h
                wCal = math.ceil(k*w)
                imgResize = cv2.resize(imgcrop,(wCal,imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:,wGap:wGap+wCal] = imgResize
            
            else:
                k = imgSize/w
                hCal = math.ceil(k*h)
                imgResize = cv2.resize(imgcrop,(imgSize,hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize-hCal)/2)
                imgWhite[hGap:hGap+hCal,:] = imgResize
        except:
            print("invalid dimension")


        try:
            #cv2.imshow("ImageCrop",imgcrop)
            cv2.imshow("Image White",imgWhite)
            #print(imgcrop.shape)
        except:
            print("invalid dimension")    
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{counter}.jpg',imgWhite)
        print(counter)