# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 07:55:04 2019

@author: vedenev
"""

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import glob

DIR = './photos_x2_cropped'

KERNEL_SIZE_GAP = 0
KERNEL_SIZE_SQUARE = 2

GRADIENT_SIZE = 1

REGION_SIZE = 4

STRIP_SIZE_RELATIVE = 1.5
CENTER_REMOVE_SIZE = 1.0

x_tmp = np.arange(-REGION_SIZE, REGION_SIZE + 1, dtype=np.float32)
X_region, Y_region = np.meshgrid(x_tmp, x_tmp)
X_region_2 = X_region **2
Y_region_2 = Y_region **2
XY_region = X_region * Y_region

def check_gradient(gradient_cut_unsigned):
    gradient_cut_unsigned_sum = np.sum(gradient_cut_unsigned)
    
    x2_mean_tmp = np.sum(X_region_2 * gradient_cut_unsigned) / gradient_cut_unsigned_sum
    #x_mean_tmp = np.sum(X_region * gradient_cut_unsigned) / gradient_cut_unsigned_sum
    y2_mean_tmp = np.sum(Y_region_2 * gradient_cut_unsigned) / gradient_cut_unsigned_sum
    xy_mean_tmp = np.sum(XY_region * gradient_cut_unsigned) / gradient_cut_unsigned_sum
    matrix_tmp = np.asarray([[x2_mean_tmp, xy_mean_tmp], 
                             [xy_mean_tmp, y2_mean_tmp]])
    values_tmp, vectors_tmp = np.linalg.eig(matrix_tmp)
    max_ind_tmp = np.argmax(values_tmp)
    main_direction_tmp = vectors_tmp[:, max_ind_tmp]
    main_direction_perpendicular_tmp = np.asarray([-main_direction_tmp[1], main_direction_tmp[0]])
    #retval, gradient_trhesholded = cv2.threshold(gradient_cut_unsigned, gradient_threshold, 255, cv2.THRESH_BINARY)
    
    projection_tmp4 = X_region * main_direction_perpendicular_tmp[0] + Y_region * main_direction_perpendicular_tmp[1]
    projection_tmp4_mean = np.sum(projection_tmp4 * gradient_cut_unsigned) / gradient_cut_unsigned_sum
    projection_tmp4_2_mean = np.sum(projection_tmp4**2 * gradient_cut_unsigned) / gradient_cut_unsigned_sum
    std_tmp2 = np.sqrt(projection_tmp4_2_mean - projection_tmp4_mean**2)

    strip_size = STRIP_SIZE_RELATIVE * std_tmp2
    projection_tmp5 = X_region * main_direction_tmp[0] + Y_region * main_direction_tmp[1]
    y_tmp5, x_tmp5 = np.where(np.logical_and(np.abs(projection_tmp4) <= strip_size, np.abs(projection_tmp5) >= CENTER_REMOVE_SIZE))
    #y_tmp5_positive, x_tmp5_positive = np.where(np.logical_and(np.abs(projection_tmp4) <= strip_size, projection_tmp5 >= CENTER_REMOVE_SIZE))
    #y_tmp5_negative, x_tmp5_negative = np.where(np.logical_and(np.abs(projection_tmp4) <= strip_size, projection_tmp5 <= -CENTER_REMOVE_SIZE))
    gradient_on_strip = gradient_cut_unsigned[y_tmp5, x_tmp5]
    mean_tmp6 = np.mean(gradient_on_strip)
    std_tmp6 = np.std(gradient_on_strip)
    flatness_tmp6 = std_tmp6 / mean_tmp6

        
    return std_tmp2, flatness_tmp6

kernel_size = 2 * KERNEL_SIZE_GAP + 1 + 2 * KERNEL_SIZE_SQUARE
kernel = np.zeros((kernel_size, kernel_size), np.float32) 
kernel[0: KERNEL_SIZE_SQUARE, 0: KERNEL_SIZE_SQUARE] = -1.0
kernel[0: KERNEL_SIZE_SQUARE, -KERNEL_SIZE_SQUARE:] = 1.0
kernel[-KERNEL_SIZE_SQUARE:, 0: KERNEL_SIZE_SQUARE] = 1.0
kernel[-KERNEL_SIZE_SQUARE:, -KERNEL_SIZE_SQUARE:] = -1.0

#frame = cv2.imread('example_1280x1024.png')
#frame = cv2.imread('photos_x2_cropped/00250.png')
#frame = cv2.imread('photos_x2_cropped/00450.png')
#frame = cv2.imread('photos_x2_cropped/00600.png')

gradient_size_full = 2 * GRADIENT_SIZE + 1

kernel_gradient_x = np.zeros((1,gradient_size_full), np.float32)
kernel_gradient_x[0, 0] = -1.0
kernel_gradient_x[0, -1] = 1.0

kernel_gradient_y = np.zeros((gradient_size_full, 1), np.float32)
kernel_gradient_y[0, 0] = -1.0
kernel_gradient_y[-1, 0] = 1.0

files = glob.glob(DIR + '/*.png')

max_all = np.zeros(len(files), np.float32)
std_x_all = np.zeros(len(files), np.float32)
flattness_x_all = np.zeros(len(files), np.float32)
std_y_all = np.zeros(len(files), np.float32)
flattness_y_all = np.zeros(len(files), np.float32)
for file_count in range(len(files)):
    frame = cv2.imread(files[file_count])
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    image_filtered = cv2.filter2D(image_gray, cv2.CV_32F, kernel)
    image_filtered_abs = np.abs(image_filtered)
    
    y_max, x_max = np.unravel_index(np.argmax(image_filtered_abs), image_filtered_abs.shape)
    max_tmp = image_filtered_abs[y_max, x_max]
    max_all[file_count] = max_tmp
    
    #plt.figure()    
    #plt.imshow(image_filtered_abs)                        
    #plt.plot(x_max, y_max, 'w.')
    #plt.colorbar()
    
    max_y, max_x = y_max, x_max
    
    sign = np.sign(image_filtered[max_y, max_x])
            
    xx1 = max_x - REGION_SIZE - 1
    xx2 = max_x + REGION_SIZE + 1
    yy1 = max_y - REGION_SIZE - 1
    yy2 = max_y + REGION_SIZE + 1
    
    image_gray_cut = image_gray[yy1: yy2 + 1, xx1: xx2 + 1]
    
    gradient_x_cut = cv2.filter2D(image_gray_cut, cv2.CV_32F, kernel_gradient_x)[1:-1, 1:-1]
    gradient_y_cut = cv2.filter2D(image_gray_cut, cv2.CV_32F, kernel_gradient_y)[1:-1, 1:-1]
    
    #gradient_threshold = GRADIENT_TRHESHOLD_RELATIVE * image_filtered_abs[max_y, max_x] / kernel_size_square_2
    
    gradient_x_cut_unsigned = np.copy(gradient_x_cut)
    gradient_x_cut_unsigned[0:REGION_SIZE,:] = sign * gradient_x_cut_unsigned[0:REGION_SIZE,:]
    gradient_x_cut_unsigned[-REGION_SIZE:,:] = -sign * gradient_x_cut_unsigned[-REGION_SIZE:,:]
    gradient_x_cut_unsigned[gradient_x_cut_unsigned < 0.0] = 0.0
    std_x, flattness_x = check_gradient(gradient_x_cut_unsigned)

    gradient_y_cut_unsigned = np.copy(gradient_y_cut)
    gradient_y_cut_unsigned[:, 0:REGION_SIZE] = sign * gradient_y_cut_unsigned[:, 0:REGION_SIZE]
    gradient_y_cut_unsigned[:, -REGION_SIZE:] = -sign * gradient_y_cut_unsigned[:, -REGION_SIZE:]
    gradient_y_cut_unsigned[gradient_y_cut_unsigned < 0.0] = 0.0
    std_y, flattness_y = check_gradient(gradient_y_cut_unsigned)
    
    std_x_all[file_count] = std_x
    flattness_x_all[file_count] = flattness_x
    std_y_all[file_count] = std_y
    flattness_y_all[file_count] = flattness_y
    

#plt.close('all')


plt.subplot(3,2,1)    
plt.plot(max_all, 'k.-')
plt.title('max')


plt.subplot(3,2,3)
plt.plot(std_x_all, 'k.-')
plt.title('std x')

plt.subplot(3,2,5)
plt.plot(flattness_x_all, 'k.-')
plt.title('flatness x')


plt.subplot(3,2,4)
plt.plot(std_y_all, 'k.-')
plt.title('std y')

plt.subplot(3,2,6)
plt.plot(flattness_y_all, 'k.-')
plt.title('flatness y')

print('np.min(max_all) =', np.min(max_all))
print('max(np.max(std_x_all), np.max(std_y_all)) =', max(np.max(std_x_all), np.max(std_y_all)))
print('max(np.max(flattness_x_all), np.max(flattness_x_all)) =', max(np.max(flattness_x_all), np.max(flattness_x_all)))