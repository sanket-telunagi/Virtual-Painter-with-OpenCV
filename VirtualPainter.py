import cv2
import numpy as np
import time
import os
from cvzone import HandTrackingModule as htm
# from cvzone import PoseModule as pos


folderPath = "Assets"
mylist = os.listdir(folderPath)
print(mylist)

overlayList = []
for imPath in mylist :
    img = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(img)

header = overlayList[0]
print(header.shape)
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.HandDetector(detectionCon=0.85)
# fingerPos = pos.PoseDetector()

counter = 0
while counter!=10 :
    # Import image
    success, img = cap.read()

    # flipping the img
    img = cv2.flip(img,1)

    # hand landmarks
    _ , img= detector.findHands(img,flipType=False)
    
    lmList= findPosition(img,draw=False)

    if len(lmList) != 0 :
        print(lmList)

        # tip of index finger and middle finger
        x1, y1,_ = lmList[8][1:]
        x2,y2,_ = lmList[12][1:]
        print(x2)

    # Determine which fingers are up

    # Two fingers up : selection mode

    # one finger up : Drawing mode

    # Resizing the image 
    

    imgs = cv2.resize(img,(1280,960))
    imgs[0:125,0:1280] = header
    cv2.imshow("Image",imgs)

    counter += 1
    print(counter)
    cv2.waitKey(1)

