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

KERNEL_SIZE_GAP = 0
KERNEL_SIZE_SQUARE = 2

GRADIENT_SIZE = 1

FILTERED_THRESHOLD = 300.0

REGION_SIZE = 3

def to_limits(x1, x2, y1, y2, shape):
    x_lim = shape[1] - 1
    y_lim = shape[0] - 1
    if x1 < 0:
        x1 = 0
    if x1 > x_lim:
        x1 = x_lim
    
    if y1 < 0:
        y1 = 0
    if y1 > y_lim:
        y1 = y_lim
    
    return x1, x2, y1, y2
        

image = cv2.imread(FILENAME)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel_size = 2 * KERNEL_SIZE_GAP + 1 + 2 * KERNEL_SIZE_SQUARE
kernel = np.zeros((kernel_size, kernel_size), np.float32) 
kernel[0: KERNEL_SIZE_SQUARE, 0: KERNEL_SIZE_SQUARE] = -1.0
kernel[0: KERNEL_SIZE_SQUARE, -KERNEL_SIZE_SQUARE:] = 1.0
kernel[-KERNEL_SIZE_SQUARE:, 0: KERNEL_SIZE_SQUARE] = 1.0
kernel[-KERNEL_SIZE_SQUARE:, -KERNEL_SIZE_SQUARE:] = -1.0

gradient_size_full = 2 * GRADIENT_SIZE + 1

kernel_gradient_x = np.zeros((1,gradient_size_full), np.float32)
kernel_gradient_x[0, 0] = -1.0
kernel_gradient_x[0, -1] = 1.0

kernel_gradient_y = np.zeros((gradient_size_full, 1), np.float32)
kernel_gradient_y[0, 0] = -1.0
kernel_gradient_y[-1, 0] = 1.0

image_filtered = cv2.filter2D(image_gray, cv2.CV_32F, kernel)
image_filtered_abs = np.abs(image_filtered)
retval, image_thresholded = cv2.threshold(image_filtered_abs, FILTERED_THRESHOLD, 255, cv2.THRESH_BINARY)
image_thresholded = image_thresholded.astype(np.uint8)
#image_thresholded =  >= FILTERED_THRESHOLD

#y_selected, x_selected = np.where(image_thresholded)
#for selected_count in range

#n_labels, labels = cv2.connectedComponents(image_thresholded, connectivity=4)
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_thresholded, connectivity=4)

for component_count in range(1, n_labels):
    x1 = stats[component_count, cv2.CC_STAT_LEFT]
    y1 = stats[component_count, cv2.CC_STAT_TOP]
    height_tmp = stats[component_count, cv2.CC_STAT_HEIGHT]
    width_tmp = stats[component_count, cv2.CC_STAT_WIDTH]
    y2 = y1 + height_tmp
    x2 = x1 + width_tmp
    image_cut = image_filtered_abs[y1:y2, x1:x2]
    max_y_inside, max_x_inside = np.unravel_index(image_cut.argmax(), image_cut.shape)
    max_y = y1 + max_y_inside
    max_x = x1 + max_x_inside
    #sign = np.sign(image_filtered[max_y, max_x])
    
    xx1 = max_x - REGION_SIZE
    xx2 = max_x + REGION_SIZE
    yy1 = max_y - REGION_SIZE
    yy2 = max_y + REGION_SIZE
    
    xx1, xx2, yy1, yy2 = to_limits(xx1, xx2, yy1, yy2, image_gray.shape)
    
    
    
    
    
    

plt.subplot(2,1,1)
plt.imshow(image_gray, cmap='gray', vmin=0, vmax=255)

plt.subplot(2,1,2)
plt.imshow(image_thresholded)
plt.colorbar()