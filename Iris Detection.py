"""
Created on Fri May  1 21:54:29 2020

@author: The Rat Pack
"""
import numpy as np
import cv2
from time import sleep

#loading the classifier
eye_cascade = cv2.CascadeClassifier(r'C:\Users\amrem\anaconda3\Lib\site-packages\cv2\data\haarcascade_eye.xml')
#loading the camera using the defult camera
camera = cv2.VideoCapture(0)
#numerator and denominator is used to to get accuracy
numerator = 0
denominator = 0
detect_counter = 0
#detection = 1 in case of detection of iris 
detection = 0
#counter is used to make a delay effect 
counter = 0
while True:
    #in case of detection
    if detection:
        detect_counter +=1
        if detect_counter > 4:
            counter = 0
            print("Iris Detected")
            detect_counter = 0
        
        
    #in case if counter > NUM then the iris wasn't detected for a period of time
    if counter > 20:
        print("WARNING")
    
    #reseting the detection flag
    detection = 0
    #get the currect frame
    ret, frame = camera.read()
    roi = frame
    
    #flip the frame around Y-Axis
    frame = cv2.flip(frame, 1)
    
    #convert current frame  to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #detect the eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    #in eyes = emptey (no eyes detected)
    if len(eyes) == 0:
        counter +=1
        #detection = 1
    for (x, y, w, h) in eyes:
        #draw a rectangle around the eyes in blue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)

        #coordinates of the iris (found by experement)
        x1 = int(x + w / 2) + 1  
        x2 = int(x + w / 1.5)
        y1 = int(y + h / 2) + 1
        y2 = int(y + h / 1.5)
        
        # drawing a rectangle around the iris
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        #using the area of the iris as the ROI
        roi = frame[y1:y2, x1:x2]
        
        #convirting the ROI into grey
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        #using Histogram Equalization to improves the contrast in an image, in order to stretch out the intensity range
        equ = cv2.equalizeHist(gray)
        
        #convirting image int binary image
        thres = cv2.inRange(equ, 0, 20)
        
        #setting up dialation and erosion parametars
        kernel = np.ones((3, 3), np.uint8)
        
        #removing small noise inside the white image
        dilation = cv2.dilate(thres, kernel, iterations=2)
        
        #decreasing the size of the white region
        erosion = cv2.erode(dilation, kernel, iterations=3)
        
        #finding the contours of the iris
        contours, hierarchy = cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        #checking for 2 contours found or not
        if len(contours) == 2:
            detection = 1
            numerator += 1
            # img = cv2.drawContours(roi, contours, 1, (0,255,0), 3)
            #finding the centroid of the contour
            M = cv2.moments(contours[1])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.line(roi, (cx, cy), (cx, cy), (0, 0, 255), 3)
                
        #checking for one countor presence
        elif len(contours) == 1:
            detection = 1
            numerator += 1
            # img = cv2.drawContours(roi, contours, 0, (0,255,0), 3)
            #finding centroid of the countor
            M = cv2.moments(contours[0])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.line(roi, (cx, cy), (cx, cy), (0, 0, 255), 3)
        #found no countor
        else:
            denominator += 1



        ran = x2 - x1
        mid = ran / 2

    cv2.imshow("frame", frame)
    # cv2.imshow("eye",image)
    if cv2.waitKey(30) == 27 & 0xff:
        break
camera.release()
print("accurracy="), (float(numerator) / float(numerator + denominator)) * 100
cv2.destroyAllWindows()