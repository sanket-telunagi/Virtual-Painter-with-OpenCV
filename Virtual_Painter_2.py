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

draw_color = (0,0,255)  # drawing color in BGR

# brush size
brush_size = 25
eraser_size = 50

overlay_list = []

for imPath in mylist :
    img = cv2.imread(f'{folder_path}/{imPath}')
    overlay_list.append(img)

header = overlay_list[0]

# previous points 
xp , yp = 0, 0

# empty image canvas of same dimentions and having values of 0-255 (8-bit)
img_canvas = np.zeros((960,1280,3), np.uint8) 

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)



with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence = 0.5) as hands :
    counter = 0
    while cap.isOpened() and counter != 500:
        ret,frame = cap.read()

        # flipping image
        frame = cv2.flip(frame,1)

        # BGR to RGB
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        # Resize
        imgs = cv2.resize(img,(1280,960))

        # Set flag
        imgs.flags.writeable = False

        # Detections of hands
        results = hands.process(imgs)

        # set flag to true 
        imgs.flags.writeable = True

        # RGB 2 BGR
        imgs = cv2.cvtColor(imgs,cv2.COLOR_RGB2BGR)

        lmList = findPosition(imgs, results, draw=False)

        # index and middle finger landmarks 
        if len(lmList) != 0 :

            xp, yp = 0, 0 

            # tip of index and middle finger
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]

            # check which fingers are up
            fingers = fingerUp(lmList,results)
            print(fingers)

            # selecting a mode

            # 1. selection mode : two fingers are up
            if fingers[1] and fingers[2] :

                
                # selecting color 
                if y1 < 125 :
                    print(f"({x1},{y1})")

                    # red
                    if 200 < x1 < 380 :
                        draw_color = (0,0,255)
                        header = overlay_list[0]
                    
                    # blue
                    elif 420 < x1 < 600 :
                        draw_color = (255,0,0)
                        header = overlay_list[1]
                    
                    # yellow
                    elif 680 < x1 < 800 :
                        draw_color = (0,255,255)
                        header = overlay_list[2]
                    
                    # eraser
                    elif 950 < x1 < 1200 :
                        draw_color = (255,255,255) # white 
                        header = overlay_list[3]
                
                # Draw a rectangle as an indication
                cv2.rectangle(imgs, (x1,y1 - 25), (x2,y2 + 25), draw_color, cv2.FILLED) 
                print("selection mode")

            # 2. Drawing mode : only index finger is up
            elif fingers[1] and fingers[2] == 0 :

                # draw circle as an indication
                cv2.circle(imgs,(x1,y1), 25, draw_color, cv2.FILLED)

                if xp | yp == 0 :
                    xp , yp = x1,y1
                
                # erasing
                if draw_color == (255,255,255) :
                    cv2.line(imgs, (xp,yp), draw_color, eraser_size)
                    cv2.line(img_canvas, (xp,yp), draw_color, eraser_size)
                
                else :
                    cv2.line(imgs, (xp,yp),(x1,y1) ,draw_color, eraser_size)
                    cv2.line(img_canvas, (xp,yp), (x1,y1), draw_color, brush_size)
                print("Drawing mode")
                xp , yp = x1, y1


        # Adding the canvas ans image
        # convert to gray
        img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)

        # inverting image
        _,img_invert = cv2.threshold(img_gray, 50,255, cv2.THRESH_BINARY_INV)

        # converting inverted image to color
        img_invert = cv2.cvtColor(img_invert, cv2.COLOR_GRAY2BGR)
        print(img_invert.shape, imgs.shape)
        # anding images
        imgs = cv2.bitwise_and(imgs, img_invert)

        # adding colors
        imgs = cv2.bitwise_or(imgs,img_canvas)

        # Rendering results on image
        if results.multi_hand_landmarks:
            for num,hand in enumerate(results.multi_hand_landmarks) :
                mp_drawing.draw_landmarks(imgs,hand,mp_hands.HAND_CONNECTIONS , 
                    mp_drawing.DrawingSpec(color=(0,230,20),thickness=2, circle_radius=2) ,  
                        mp_drawing.DrawingSpec(color=(0,0,230),thickness=1, circle_radius=1))

        # Showing image
        
        imgs[0:125,0:1280] = header
        cv2.imshow("Hand Tracking",imgs)
        cv2.imshow("canvas",img_canvas)
        if cv2.waitKey(10) & 0xFF == ord('q') :
            break
        counter += 1

cap.release()
cv2.destroyAllWindows()