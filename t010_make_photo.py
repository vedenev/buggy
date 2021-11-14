# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:09:14 2019

@author: vedenev
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def imshow_bgr(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2 * 480)

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

offsset = 10
count = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        
        count += 1
        #if count >= offsset:
        #    break

imshow_bgr(frame)
cv2.imwrite('./photo_tmp.png', frame)
cap.release()
    