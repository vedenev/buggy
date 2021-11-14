# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 07:55:07 2019

@author: vedenev
"""

import cv2
import matplotlib.pyplot as plt 
import numpy as np

#FILENAME = './2019_11_26_07_54.jpg'
#FILENAME = './2019_11_27_06_51.jpg'
FILENAME = './2019_11_27_07_12.jpg'
#FILENAME = './2019_11_27_08_12_orig.jpg'
image = cv2.imread(FILENAME)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#kernel = np.asarray([[1, -1],
#                     [-1, 1]], dtype=np.float32)

kernel = np.zeros((6,6), np.float32) 
kernel[0:2, 0:2] = -1
kernel[0:2, 4:6] = 1
kernel[4:6, 0:2] = 1
kernel[4:6, 4:6] = -1

#kernel = np.zeros((5,5), np.float32) 
#kernel[0:2, 0:2] = -1
#kernel[0:2, 3:5] = 1
#kernel[3:5, 0:2] = 1
#kernel[3:5, 3:5] = -1

response_tmp1 = cv2.filter2D(image_gray, cv2.CV_32F, kernel)

plt.subplot(3,1,1)
plt.imshow(image_gray, cmap='gray', vmin=0, vmax=255)
plt.subplot(3,1,2)
plt.imshow(response_tmp1)
plt.colorbar()
plt.subplot(3,1,3)
plt.imshow(response_tmp1 >= 600)
