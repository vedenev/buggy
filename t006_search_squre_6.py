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

#FILTERED_THRESHOLD = 300.0
FILTERED_THRESHOLD = 500.0

REGION_SIZE = 4
GRADIENT_TRHESHOLD_RELATIVE = 0.5
N_POINTS_TRESHOLD_RELATIVE = 0.7
STD_TRESHOLD = 0.7

kernel_size_square_2 = KERNEL_SIZE_SQUARE ** 2

n_points_threshold = int(np.round(N_POINTS_TRESHOLD_RELATIVE * 2 * REGION_SIZE))

x_tmp = np.arange(-REGION_SIZE, REGION_SIZE + 1, dtype=np.float32)
X_region, Y_region = np.meshgrid(x_tmp, x_tmp)
X_region_2 = X_region **2
Y_region_2 = Y_region **2
XY_region = X_region * Y_region

def check_gradient(gradient_cut_unsigned, gradient_threshold):
    gradient_cut_unsigned_sum = np.sum(gradient_cut_unsigned)
    
    x2_mean_tmp = np.sum(X_region_2 * gradient_cut_unsigned) / gradient_cut_unsigned_sum
    y2_mean_tmp = np.sum(Y_region_2 * gradient_cut_unsigned) / gradient_cut_unsigned_sum
    xy_mean_tmp = np.sum(XY_region * gradient_cut_unsigned) / gradient_cut_unsigned_sum
    matrix_tmp = np.asarray([[x2_mean_tmp, xy_mean_tmp], 
                             [xy_mean_tmp, y2_mean_tmp]])
    values_tmp, vectors_tmp = np.linalg.eig(matrix_tmp)
    max_ind_tmp = np.argmax(values_tmp)
    main_direction_tmp = vectors_tmp[:, max_ind_tmp]
    main_direction_perpendicular_tmp = np.asarray([-main_direction_tmp[1], main_direction_tmp[0]])
    retval, gradient_trhesholded = cv2.threshold(gradient_cut_unsigned, gradient_threshold, 255, cv2.THRESH_BINARY)
    
    y_tmp2, x_tmp2 = np.where(gradient_trhesholded > 0)
    is_gradient_ok = False
    if y_tmp2.size >= n_points_threshold:
        projection_tmp2 = x_tmp2 * main_direction_perpendicular_tmp[0] + y_tmp2 * main_direction_perpendicular_tmp[1]
        std_tmp2 = np.std(projection_tmp2)
        if std_tmp2 <= STD_TRESHOLD:
            is_gradient_ok = True
    return is_gradient_ok

image = cv2.imread(FILENAME)



image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#image_gray = np.rot90(image_gray)

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

gradient_x = cv2.filter2D(image_gray, cv2.CV_32F, kernel_gradient_x)
gradient_y = cv2.filter2D(image_gray, cv2.CV_32F, kernel_gradient_y)

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
    sign = np.sign(image_filtered[max_y, max_x])
    
    xx1 = max_x - REGION_SIZE
    xx2 = max_x + REGION_SIZE
    yy1 = max_y - REGION_SIZE
    yy2 = max_y + REGION_SIZE
    
    
    if 0 <= xx1  and xx2 < image_gray.shape[1] and 0 <= yy1  and yy2 < image_gray.shape[0]:
        gradient_x_cut = gradient_x[yy1: yy2 + 1, xx1: xx2 + 1]
        gradient_y_cut = gradient_y[yy1: yy2 + 1, xx1: xx2 + 1]
        
        gradient_threshold = GRADIENT_TRHESHOLD_RELATIVE * image_filtered_abs[max_y, max_x] / kernel_size_square_2
        
        
        gradient_x_cut_unsigned = np.copy(gradient_x_cut)
        gradient_x_cut_unsigned[0:REGION_SIZE,:] = sign * gradient_x_cut_unsigned[0:REGION_SIZE,:]
        gradient_x_cut_unsigned[-REGION_SIZE:,:] = -sign * gradient_x_cut_unsigned[-REGION_SIZE:,:]
        gradient_x_cut_unsigned[gradient_x_cut_unsigned < 0.0] = 0.0
        is_x_gradient_ok = check_gradient(gradient_x_cut_unsigned, gradient_threshold)
        
        gradient_y_cut_unsigned = np.copy(gradient_y_cut)
        gradient_y_cut_unsigned[:, 0:REGION_SIZE] = sign * gradient_y_cut_unsigned[:, 0:REGION_SIZE]
        gradient_y_cut_unsigned[:, -REGION_SIZE:] = -sign * gradient_y_cut_unsigned[:, -REGION_SIZE:]
        gradient_y_cut_unsigned[gradient_y_cut_unsigned < 0.0] = 0.0
        is_y_gradient_ok = check_gradient(gradient_y_cut_unsigned, gradient_threshold)
                
                
    
    
    
    

plt.subplot(2,1,1)
plt.imshow(image_gray, cmap='gray', vmin=0, vmax=255)

plt.subplot(2,1,2)
plt.imshow(image_thresholded)
plt.colorbar()