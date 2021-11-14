# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 07:26:21 2019

@author: vedenev
"""

import cv2
import time
import numpy as np

imageName = 'last_frrame.jpg' 

KERNEL_SIZE_GAP = 0
KERNEL_SIZE_SQUARE = 2

GRADIENT_SIZE = 1

FILTERED_THRESHOLD = 300.0
#FILTERED_THRESHOLD = 500.0

REGION_SIZE = 4
GRADIENT_TRHESHOLD_RELATIVE = 0.3
N_POINTS_TRESHOLD_RELATIVE = 0.6
N_POINTS_ONE_SIDE_TRESHOLD_RELATIVE = 0.125

STD_TRESHOLD = 1.5
FLATNESS_TRHESHOLD = 0.7

CROSS_TIP_THRESHOLD = 1

STRIP_SIZE_RELATIVE = 1.5
CENTER_REMOVE_SIZE = 1.0

kernel_size_square_2 = KERNEL_SIZE_SQUARE ** 2

n_points_threshold = int(np.round(N_POINTS_TRESHOLD_RELATIVE * 2 * REGION_SIZE))
n_points_one_side_threshold = int(np.round(N_POINTS_ONE_SIDE_TRESHOLD_RELATIVE * 2 * REGION_SIZE))

x_tmp = np.arange(-REGION_SIZE, REGION_SIZE + 1, dtype=np.float32)
X_region, Y_region = np.meshgrid(x_tmp, x_tmp)
X_region_2 = X_region **2
Y_region_2 = Y_region **2
XY_region = X_region * Y_region

kernel_size = 2 * KERNEL_SIZE_GAP + 1 + 2 * KERNEL_SIZE_SQUARE
kernel_size_2 = (kernel_size - 1) // 2
kernel = np.zeros((kernel_size, kernel_size), np.float32) 
kernel[0: KERNEL_SIZE_SQUARE, 0: KERNEL_SIZE_SQUARE] = -1.0
kernel[0: KERNEL_SIZE_SQUARE, -KERNEL_SIZE_SQUARE:] = 1.0
kernel[-KERNEL_SIZE_SQUARE:, 0: KERNEL_SIZE_SQUARE] = 1.0
kernel[-KERNEL_SIZE_SQUARE:, -KERNEL_SIZE_SQUARE:] = -1.0

#kernel_points = np.zeros((4 * 4, 2), np.int32)
#kernel_points_signs = np.zeros(4 * 4, np.int32)
#
#kernel_points[0, 0] = 0
#kernel_points[0, 1] = 0
#kernel_points_signs[0] = 1
#
#kernel_points[1, 0] = KERNEL_SIZE_SQUARE
#kernel_points[1, 1] = KERNEL_SIZE_SQUARE
#kernel_points_signs[0] = 1
#
#kernel_points[2, 0] = 0
#kernel_points[2, 1] = KERNEL_SIZE_SQUARE
#kernel_points_signs[0] = -1
#
#kernel_points[3, 0] = KERNEL_SIZE_SQUARE
#kernel_points[3, 1] = 0
#kernel_points_signs[0] = -1



gradient_size_full = 2 * GRADIENT_SIZE + 1

kernel_gradient_x = np.zeros((1,gradient_size_full), np.float32)
kernel_gradient_x[0, 0] = -1.0
kernel_gradient_x[0, -1] = 1.0

kernel_gradient_y = np.zeros((gradient_size_full, 1), np.float32)
kernel_gradient_y[0, 0] = -1.0
kernel_gradient_y[-1, 0] = 1.0

def check_gradient(gradient_cut_unsigned, gradient_threshold):
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
    is_gradient_ok = False
    #is_gradient_ok = True
    if std_tmp2 <= STD_TRESHOLD:
    #if True:
        strip_size = STRIP_SIZE_RELATIVE * std_tmp2
        projection_tmp5 = X_region * main_direction_tmp[0] + Y_region * main_direction_tmp[1]
        y_tmp5, x_tmp5 = np.where(np.logical_and(np.abs(projection_tmp4) <= strip_size, np.abs(projection_tmp5) >= CENTER_REMOVE_SIZE))
        #y_tmp5_positive, x_tmp5_positive = np.where(np.logical_and(np.abs(projection_tmp4) <= strip_size, projection_tmp5 >= CENTER_REMOVE_SIZE))
        #y_tmp5_negative, x_tmp5_negative = np.where(np.logical_and(np.abs(projection_tmp4) <= strip_size, projection_tmp5 <= -CENTER_REMOVE_SIZE))
        gradient_on_strip = gradient_cut_unsigned[y_tmp5, x_tmp5]
        mean_tmp6 = np.mean(gradient_on_strip)
        std_tmp6 = np.std(gradient_on_strip)
        flatness_tmp6 = std_tmp6 / mean_tmp6
        if flatness_tmp6 <= FLATNESS_TRHESHOLD:
            is_gradient_ok = True
            print("std_tmp2 =", std_tmp2)
            print("flatness_tmp6 =", flatness_tmp6)
        
    return is_gradient_ok

cap = cv2.VideoCapture(0)
height = 2 * 480
width = 2 * 640
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #For capture image in monochrome
    #rgbImage = frame #For capture the image in RGB color space

    # Display the resulting frame
    if ret:
        
        frame_orig = np.copy(frame)
        
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #image_filtered = cv2.filter2D(image_gray, cv2.CV_32F, kernel)
        
        #image_integral = cv2.integral(image_gray)
        #image_filtered = np.zeros((image_gray.shape[0], image_gray.shape[1]), np.int32)
        
        #np.sum(image_gray[110:120, 110:120]) = 25500
        #image_integral[110, 110] + image_integral[120, 120] - image_integral[110, 120] - image_integral[120, 110] = 25500
        
        #kernel =
        #array([[-1., -1.,  0.,  1.,  1.],
        #       [-1., -1.,  0.,  1.,  1.],
        #       [ 0.,  0.,  0.,  0.,  0.],
        #       [ 1.,  1.,  0., -1., -1.],
        #       [ 1.,  1.,  0., -1., -1.]], dtype=float32)
        # kernel_size_2 = 2
        # kernel_size = 5
        # KERNEL_SIZE_SQUARE = 2
        
        #image_filtered[kernel_size_2: -kernel_size_2, kernel_size_2: -kernel_size_2] = image_integral[0:kernel_size, 0:kernel_size] + image_integral[kernel_size:, 0:kernel_size:]
        
        image_filtered = np.zeros((image_gray.shape[0], image_gray.shape[1]), np.int32)
        image_gray_i32 = image_gray.astype(np.int32)
        image_filtered[kernel_size_2: height - kernel_size_2, kernel_size_2: width - kernel_size_2] = +(\
        \
        - image_gray_i32[kernel_size_2 - 2: height - kernel_size_2 - 2, kernel_size_2 - 2: width - kernel_size_2 - 2] \
        - image_gray_i32[kernel_size_2 - 2: height - kernel_size_2 - 2, kernel_size_2 - 1: width - kernel_size_2 - 1] \
        - image_gray_i32[kernel_size_2 - 1: height - kernel_size_2 - 1, kernel_size_2 - 2: width - kernel_size_2 - 2] \
        - image_gray_i32[kernel_size_2 - 1: height - kernel_size_2 - 1, kernel_size_2 - 1: width - kernel_size_2 - 1] \
        \
        + image_gray_i32[kernel_size_2 - 1: height - kernel_size_2 - 1, kernel_size_2 + 1: width - kernel_size_2 + 1] \
        + image_gray_i32[kernel_size_2 - 1: height - kernel_size_2 - 1, kernel_size_2 + 2: width - kernel_size_2 + 2] \
        + image_gray_i32[kernel_size_2 - 2: height - kernel_size_2 - 2, kernel_size_2 + 1: width - kernel_size_2 + 1] \
        + image_gray_i32[kernel_size_2 - 2: height - kernel_size_2 - 2, kernel_size_2 + 2: width - kernel_size_2 + 2] \
        \
        + image_gray_i32[kernel_size_2 + 1: height - kernel_size_2 + 1, kernel_size_2 - 2: width - kernel_size_2 - 2] \
        + image_gray_i32[kernel_size_2 + 1: height - kernel_size_2 + 1, kernel_size_2 - 1: width - kernel_size_2 - 1] \
        + image_gray_i32[kernel_size_2 + 2: height - kernel_size_2 + 2, kernel_size_2 - 2: width - kernel_size_2 - 2] \
        + image_gray_i32[kernel_size_2 + 2: height - kernel_size_2 + 2, kernel_size_2 - 1: width - kernel_size_2 - 1] \
        \
        - image_gray_i32[kernel_size_2 + 1: height - kernel_size_2 + 1, kernel_size_2 + 1: width - kernel_size_2 + 1] \
        - image_gray_i32[kernel_size_2 + 1: height - kernel_size_2 + 1, kernel_size_2 + 2: width - kernel_size_2 + 2] \
        - image_gray_i32[kernel_size_2 + 2: height - kernel_size_2 + 2, kernel_size_2 + 1: width - kernel_size_2 + 1] \
        - image_gray_i32[kernel_size_2 + 2: height - kernel_size_2 + 2, kernel_size_2 + 2: width - kernel_size_2 + 2] )
        
        
        
        
        image_filtered_abs = np.abs(image_filtered).astype(np.float32)
        retval, image_thresholded = cv2.threshold(image_filtered_abs, FILTERED_THRESHOLD, 255, cv2.THRESH_BINARY)
        image_thresholded = image_thresholded.astype(np.uint8)

        
        #gradient_x = cv2.filter2D(image_gray, cv2.CV_32F, kernel_gradient_x)
        #gradient_y = cv2.filter2D(image_gray, cv2.CV_32F, kernel_gradient_y)
        
        #n_labels, labels = cv2.connectedComponents(image_thresholded, connectivity=4)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_thresholded, connectivity=4)
        
        x_tmp = 0
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
            
            xx1 = max_x - REGION_SIZE - 1
            xx2 = max_x + REGION_SIZE + 1
            yy1 = max_y - REGION_SIZE - 1
            yy2 = max_y + REGION_SIZE + 1
            
            
            if 0 <= xx1  and xx2 < image_gray.shape[1] and 0 <= yy1  and yy2 < image_gray.shape[0]:
                #gradient_x_cut = gradient_x[yy1: yy2 + 1, xx1: xx2 + 1]
                #gradient_y_cut = gradient_y[yy1: yy2 + 1, xx1: xx2 + 1]
                
                image_gray_cut = image_gray[yy1: yy2 + 1, xx1: xx2 + 1]
                
                gradient_x_cut = cv2.filter2D(image_gray_cut, cv2.CV_32F, kernel_gradient_x)[1:-1, 1:-1]
                gradient_y_cut = cv2.filter2D(image_gray_cut, cv2.CV_32F, kernel_gradient_y)[1:-1, 1:-1]
                
                gradient_threshold = GRADIENT_TRHESHOLD_RELATIVE * image_filtered_abs[max_y, max_x] / kernel_size_square_2
                
                
                #print(' ')
                #print(' ')
                #print('x')
                gradient_x_cut_unsigned = np.copy(gradient_x_cut)
                gradient_x_cut_unsigned[0:REGION_SIZE,:] = sign * gradient_x_cut_unsigned[0:REGION_SIZE,:]
                gradient_x_cut_unsigned[-REGION_SIZE:,:] = -sign * gradient_x_cut_unsigned[-REGION_SIZE:,:]
                gradient_x_cut_unsigned[gradient_x_cut_unsigned < 0.0] = 0.0
                is_x_gradient_ok = check_gradient(gradient_x_cut_unsigned, gradient_threshold)
                #print(' ')
                #print('y')
                gradient_y_cut_unsigned = np.copy(gradient_y_cut)
                gradient_y_cut_unsigned[:, 0:REGION_SIZE] = sign * gradient_y_cut_unsigned[:, 0:REGION_SIZE]
                gradient_y_cut_unsigned[:, -REGION_SIZE:] = -sign * gradient_y_cut_unsigned[:, -REGION_SIZE:]
                gradient_y_cut_unsigned[gradient_y_cut_unsigned < 0.0] = 0.0
                is_y_gradient_ok = check_gradient(gradient_y_cut_unsigned, gradient_threshold)

                if is_x_gradient_ok and is_y_gradient_ok:
                    frame = cv2.circle(frame, (max_x, max_y), 2, (0, 0, 255), -1)
                    
                    img_tmp2 = (255.0 * gradient_x_cut_unsigned / np.max(gradient_x_cut_unsigned)).astype(np.uint8)
                    img_y_tmp2 = (255.0 * gradient_y_cut_unsigned / np.max(gradient_y_cut_unsigned)).astype(np.uint8)
                    
                    #img_tmp2 = gradient_x_trhesholded
                    #img_y_tmp2 = gradient_y_trhesholded
                    
                    resize_factor_tmp = 3
                    img_tmp3 = cv2.resize(img_tmp2, (img_tmp2.shape[1] * resize_factor_tmp, img_tmp2.shape[0] * resize_factor_tmp), interpolation=cv2.INTER_NEAREST)
                    img_tmp4 = cv2.cvtColor(img_tmp3, cv2.COLOR_GRAY2BGR)
                    
                    img_y_tmp3 = cv2.resize(img_y_tmp2, (img_tmp2.shape[1] * resize_factor_tmp, img_tmp2.shape[0] * resize_factor_tmp), interpolation=cv2.INTER_NEAREST)
                    img_y_tmp4 = cv2.cvtColor(img_y_tmp3, cv2.COLOR_GRAY2BGR)
                    
                    x_tmp2 = x_tmp + img_tmp4.shape[1]
                    if x_tmp2 <= (frame.shape[1] - 1):
                        
                        frame[0:img_tmp4.shape[0], x_tmp:x_tmp2, :] = img_tmp4 
                        
                        frame[30:30 + img_tmp4.shape[0], x_tmp:x_tmp2, :] = img_y_tmp4
                        
                        x_tmp = x_tmp2 + 3
                    
        
        resize_tmp = 1
        frame_increased = cv2.resize(frame, (frame.shape[1] * resize_tmp, frame.shape[0] * resize_tmp), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Webcam', frame_increased)
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
