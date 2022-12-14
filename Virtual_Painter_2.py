import cv2 
import numpy as np
import os
import mediapipe as mp

def findPosition(img, results,color = (255,0, 255),handNo = 0, draw = True):
    lmList = []
    # handNo = 0 => right hand
    if results.multi_hand_landmarks:
        myHand = results.multi_hand_landmarks[handNo]
        for id, lm in enumerate(myHand.landmark) :
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
            if draw :
                cv2.circle(img,(cx,cy), 15, color, cv2.FILLED)
    
    return lmList

# detect which finger up
def fingerUp(land_mark_list,results,handNo = 0) :
    tip_IDs = [4,8,12,16,20]  # tip ids of each finger
    lm_list = land_mark_list  # landmark list
    fingers = []
    if results.multi_hand_landmarks :
        
        # thumb 
        if handNo == 0 :
            if lm_list[tip_IDs[0]][0] > lm_list[tip_IDs[0]-1][0] :
                fingers.append(1)
            else :
                fingers.append(0)
        else :
            if lm_list[tip_IDs[0]][0] < lm_list[tip_IDs[0]-1][0] :
                fingers.append(1)
            else :
                fingers.append(0) 

        # 4 fingers
        for id in range(1,5) :
            if lm_list[tip_IDs[id]][1] < lm_list[tip_IDs[id] - 2][1] :
                fingers.append(1)
            else :
                fingers.append(0)
    # return fingers list
    return fingers   
    

folder_path = "Assets"
mylist = os.listdir(folder_path)

overlay_list = []

for imPath in mylist :
    img = cv2.imread(f'{folder_path}/{imPath}')
    overlay_list.append(img)

header = overlay_list[0]


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence = 0.5) as hands :
    counter = 0
    while cap.isOpened() and counter != 200:
        ret,frame = cap.read()

        # flipping image
        frame = cv2.flip(frame,1)

        # BGR to RGB
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        # Set flag
        img.flags.writeable = False

        # Detections of hands
        results = hands.process(img)
        print(results)
        # set flag to true 
        img.flags.writeable = True

        # RGB 2 BGR
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        lmList = findPosition(img, results, draw=False)

        # index and middle finger landmarks 
        if len(lmList) != 0 :
            # tip of index and middle finger
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

        # check which fingers are up
        fingers = fingerUp(lmList,results)
        print(fingers)

        # detections 
        # print(results)

        # Rendering results on image
        if results.multi_hand_landmarks:
            for num,hand in enumerate(results.multi_hand_landmarks) :
                mp_drawing.draw_landmarks(img,hand,mp_hands.HAND_CONNECTIONS , 
                    mp_drawing.DrawingSpec(color=(0,230,20),thickness=2, circle_radius=2) ,  
                        mp_drawing.DrawingSpec(color=(0,0,230),thickness=1, circle_radius=1))

        # Showing image
        imgs = cv2.resize(img,(1280,960))
        imgs[0:125,0:1280] = header
        cv2.imshow("Hand Tracking",imgs)

        if cv2.waitKey(10) & 0xFF == ord('q') :
            break
        counter += 1

cap.release()
cv2.destroyAllWindows()