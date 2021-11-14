# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 07:55:04 2019

@author: vedenev
"""

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

KERNEL_SIZE_GAP = 0
KERNEL_SIZE_SQUARE = 2

kernel_size = 2 * KERNEL_SIZE_GAP + 1 + 2 * KERNEL_SIZE_SQUARE
kernel = np.zeros((kernel_size, kernel_size), np.float32) 
kernel[0: KERNEL_SIZE_SQUARE, 0: KERNEL_SIZE_SQUARE] = -1.0
kernel[0: KERNEL_SIZE_SQUARE, -KERNEL_SIZE_SQUARE:] = 1.0
kernel[-KERNEL_SIZE_SQUARE:, 0: KERNEL_SIZE_SQUARE] = 1.0
kernel[-KERNEL_SIZE_SQUARE:, -KERNEL_SIZE_SQUARE:] = -1.0

#frame = cv2.imread('example_1280x1024.png')
#frame = cv2.imread('photos_x2_cropped/00250.png')
frame = cv2.imread('photos_x2_cropped/00550.png')
#frame = cv2.imread('photos_x2_cropped/00450.png')
#frame = cv2.imread('photos_x2_cropped/00600.png')


       
image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

image_filtered = cv2.filter2D(image_gray, cv2.CV_32F, kernel)
image_filtered_abs = np.abs(image_filtered)


y_max, x_max = np.unravel_index(np.argmax(image_filtered_abs), image_filtered_abs.shape)

plt.imshow(image_filtered_abs)
plt.plot(x_max, y_max, 'w.')
plt.colorbar()