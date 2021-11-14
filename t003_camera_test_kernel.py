# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 07:26:21 2019

@author: vedenev
"""

import cv2
import time
import numpy as np

imageName = 'last_frrame.jpg' 

kernel = np.zeros((6,6), np.float32) 
kernel[0:2, 0:2] = -1
kernel[0:2, 4:6] = 1
kernel[4:6, 0:2] = 1
kernel[4:6, 4:6] = -1

thresh = 600.0

cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #For capture image in monochrome
    #rgbImage = frame #For capture the image in RGB color space

    # Display the resulting frame
    if ret:
        
        frame_orig = np.copy(frame)
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        response_tmp1 = cv2.filter2D(frame_gray, cv2.CV_32F, kernel)
        y_tmp, x_tmp = np.where(response_tmp1 > thresh)
        
        if y_tmp.size > 0:
            y_mean = np.mean(y_tmp)
            x_mean = np.mean(x_tmp)
            
            x_tmp2 = int(np.round(x_mean))
            y_tmp2 = int(np.round(y_mean))
            
            frame = cv2.circle(frame, (x_tmp2, y_tmp2), 3, (0, 0, 255), -1)
        
        cv2.imshow('Webcam', frame)
    #Wait to press 'q' key for capturing
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #Set the image name to the date it was captured
        imageName = str(time.strftime("%Y_%m_%d_%H_%M")) + '.jpg'
        #Save the image
        cv2.imwrite(imageName, frame)
        
        imageName = str(time.strftime("%Y_%m_%d_%H_%M")) + '_orig.jpg'
        cv2.imwrite(imageName, frame_orig)
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
