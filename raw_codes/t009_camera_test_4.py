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
STD_TRESHOLD = 3.5
CROSS_TIP_THRESHOLD = 1

kernel_size_square_2 = KERNEL_SIZE_SQUARE ** 2

n_points_threshold = int(np.round(N_POINTS_TRESHOLD_RELATIVE * 2 * REGION_SIZE))
n_points_one_side_threshold = int(np.round(N_POINTS_ONE_SIDE_TRESHOLD_RELATIVE * 2 * REGION_SIZE))

x_tmp = np.arange(-REGION_SIZE, REGION_SIZE + 1, dtype=np.float32)
X_region, Y_region = np.meshgrid(x_tmp, x_tmp)
X_region_2 = X_region **2
Y_region_2 = Y_region **2
XY_region = X_region * Y_region

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
    retval, gradient_trhesholded = cv2.threshold(gradient_cut_unsigned, gradient_threshold, 255, cv2.THRESH_BINARY)
    
    y_tmp2_orig, x_tmp2_orig = np.where(gradient_trhesholded > 0)
    y_tmp2 = X_region[y_tmp2_orig, x_tmp2_orig]
    x_tmp2 = Y_region[y_tmp2_orig, x_tmp2_orig]
    is_gradient_ok = False
    #is_gradient_ok = True
    if y_tmp2.size >= n_points_threshold:
        projection_tmp2 = x_tmp2 * main_direction_perpendicular_tmp[0] + y_tmp2 * main_direction_perpendicular_tmp[1]
        #std_tmp2 = np.std(projection_tmp2)
        #std_tmp2 = np.sqrt(x2_mean_tmp - x_mean_tmp**2)
        projection_tmp4 = X_region * main_direction_perpendicular_tmp[0] + Y_region * main_direction_perpendicular_tmp[1]
        projection_tmp4_mean = np.sum(projection_tmp4 * gradient_cut_unsigned) / gradient_cut_unsigned_sum
        projection_tmp4_2_mean = np.sum(projection_tmp4**2 * gradient_cut_unsigned) / gradient_cut_unsigned_sum
        std_tmp2 = np.sqrt(projection_tmp4_2_mean - projection_tmp4_mean**2)
        if std_tmp2 <= STD_TRESHOLD:
            
            projection_main_tmp2 = x_tmp2 * main_direction_tmp[0] + y_tmp2 * main_direction_tmp[1]
            
            ind_positivie_tmp3 = np.where(projection_main_tmp2 >= CROSS_TIP_THRESHOLD)[0]
            n_positive_tmp3 = ind_positivie_tmp3.size
            ind_negative_tmp3 = np.where(projection_main_tmp2 <= -CROSS_TIP_THRESHOLD)[0]
            n_negative_tmp3 = ind_negative_tmp3.size
            
            
            if n_positive_tmp3 >= n_points_one_side_threshold and n_negative_tmp3 >= n_points_one_side_threshold:
                is_gradient_ok = True
                print('std_tmp2 =', std_tmp2)
                print("n_positive_tmp3 =", n_positive_tmp3, 'n_negative_tmp3 =', n_negative_tmp3)
    return is_gradient_ok, gradient_trhesholded
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #For capture image in monochrome
    #rgbImage = frame #For capture the image in RGB color space

    # Display the resulting frame
    if ret:
        
        frame_orig = np.copy(frame)
        
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        image_filtered = cv2.filter2D(image_gray, cv2.CV_32F, kernel)
        image_filtered_abs = np.abs(image_filtered)
        retval, image_thresholded = cv2.threshold(image_filtered_abs, FILTERED_THRESHOLD, 255, cv2.THRESH_BINARY)
        image_thresholded = image_thresholded.astype(np.uint8)

        
        gradient_x = cv2.filter2D(image_gray, cv2.CV_32F, kernel_gradient_x)
        gradient_y = cv2.filter2D(image_gray, cv2.CV_32F, kernel_gradient_y)
        
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
            
            xx1 = max_x - REGION_SIZE
            xx2 = max_x + REGION_SIZE
            yy1 = max_y - REGION_SIZE
            yy2 = max_y + REGION_SIZE
            
            
            if 0 <= xx1  and xx2 < image_gray.shape[1] and 0 <= yy1  and yy2 < image_gray.shape[0]:
                gradient_x_cut = gradient_x[yy1: yy2 + 1, xx1: xx2 + 1]
                gradient_y_cut = gradient_y[yy1: yy2 + 1, xx1: xx2 + 1]
                
                gradient_threshold = GRADIENT_TRHESHOLD_RELATIVE * image_filtered_abs[max_y, max_x] / kernel_size_square_2
                
                
                print(' ')
                print(' ')
                print('x')
                gradient_x_cut_unsigned = np.copy(gradient_x_cut)
                gradient_x_cut_unsigned[0:REGION_SIZE,:] = sign * gradient_x_cut_unsigned[0:REGION_SIZE,:]
                gradient_x_cut_unsigned[-REGION_SIZE:,:] = -sign * gradient_x_cut_unsigned[-REGION_SIZE:,:]
                gradient_x_cut_unsigned[gradient_x_cut_unsigned < 0.0] = 0.0
                is_x_gradient_ok, gradient_x_trhesholded = check_gradient(gradient_x_cut_unsigned, gradient_threshold)
                print(' ')
                print('y')
                gradient_y_cut_unsigned = np.copy(gradient_y_cut)
                gradient_y_cut_unsigned[:, 0:REGION_SIZE] = sign * gradient_y_cut_unsigned[:, 0:REGION_SIZE]
                gradient_y_cut_unsigned[:, -REGION_SIZE:] = -sign * gradient_y_cut_unsigned[:, -REGION_SIZE:]
                gradient_y_cut_unsigned[gradient_y_cut_unsigned < 0.0] = 0.0
                is_y_gradient_ok, gradient_y_trhesholded = check_gradient(gradient_y_cut_unsigned, gradient_threshold)

                if is_x_gradient_ok and is_y_gradient_ok:
                    frame = cv2.circle(frame, (max_x, max_y), 2, (0, 0, 255), -1)
                    
                    #img_tmp2 = (255.0 * gradient_x_cut_unsigned / np.max(gradient_x_cut_unsigned)).astype(np.uint8)
                    #img_y_tmp2 = (255.0 * gradient_y_cut_unsigned / np.max(gradient_y_cut_unsigned)).astype(np.uint8)
                    
                    img_tmp2 = gradient_x_trhesholded
                    img_y_tmp2 = gradient_y_trhesholded
                    
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
                    
        
        resize_tmp = 2
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