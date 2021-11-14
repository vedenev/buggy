# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 07:55:07 2019

@author: vedenev
"""

import cv2
import matplotlib.pyplot as plt 
import numpy as np

#FILENAME = './2019_11_26_07_54.jpg'
FILENAME = './2019_11_27_06_51.jpg'
image = cv2.imread(FILENAME)

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

#cv2.imshow('1',image)

image_tmp = image_hsv[:, :, 1].astype(np.float32) + image_hsv[:, :, 2].astype(np.float32)

plt.imshow(image_tmp)
plt.colorbar()
