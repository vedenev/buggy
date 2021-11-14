import numpy as np
import cv2

IS_UPSIDEDOWN = True

WIDTH = 1280
HEIGHT = 960

y1_strip = 460
y2_strip = 570

KERNEL_SIZE_GAP = 0
KERNEL_SIZE_SQUARE = 2

CODE_SHAPE = (2, 7)
code_binary_length = (CODE_SHAPE[1] - 4) * CODE_SHAPE[0] + 2
code_binary_bases = 2 ** np.arange(code_binary_length - 2 - 1, -1, -1)

GRADIENT_SIZE = 1
gradient_size_full = 2 * GRADIENT_SIZE + 1
    
kernel_gradient_x = np.zeros((1,gradient_size_full), np.float32)
kernel_gradient_x[0, 0] = -1.0
kernel_gradient_x[0, -1] = 1.0

kernel_gradient_y = np.zeros((gradient_size_full, 1), np.float32)
kernel_gradient_y[0, 0] = -1.0
kernel_gradient_y[-1, 0] = 1.0


SQURE_CHECK_Y_MARGIN_2 = 4
CODE_CUTTED_UNDISTORTED_SHAPE = (100, 2* SQURE_CHECK_Y_MARGIN_2 + 1)

x = np.load('x_min_m_0_7037_2020_05_11__19_30.npy')

#FILTERED_THRESHOLD = 670.0
FILTERED_THRESHOLD = x[0] * 1300.0

#REGION_SIZE = 4
REGION_SIZE = int(np.round(8*x[1]))

#GRADIENT_TRHESHOLD_RELATIVE = 0.3
GRADIENT_TRHESHOLD_RELATIVE = 0.3 * x[2] / 0.5

#STD_TRESHOLD = 2.0
STD_TRESHOLD = 2.0 * x[3] / 0.5

#FLATNESS_TRHESHOLD = 1.0
FLATNESS_TRHESHOLD = 1.0 * x[4] / 0.5


#STRIP_SIZE_RELATIVE = 1.5
STRIP_SIZE_RELATIVE = 1.5 * x[5] / 0.5

#CENTER_REMOVE_SIZE = 1.0
CENTER_REMOVE_SIZE = 1.0 * x[6] / 0.5


#SIGNAL_STRENGTH_THRESHOLD = 10.0
SIGNAL_STRENGTH_THRESHOLD = 10.0 * x[7] / 0.5


#DIRECTIONS_ANGLE_DIFF_THRESHOLD_DEGREES = 30
DIRECTIONS_ANGLE_DIFF_THRESHOLD_DEGREES = 30 * x[8] / 0.5

#PERPENDICULAR_DISTANCE_THRESHOLD = 1.5
PERPENDICULAR_DISTANCE_THRESHOLD = 1.5 * x[9] / 0.5

#POINTS_GAP_SIZE_RELATIVE = 0.1
#POINTS_GAP_SIZE_RELATIVE = 0.1 * x[10] / 0.5
POINTS_GAP_SIZE_RELATIVE = (0.1 * x[10] / 0.5) * 1.3 # increase after optimization
# 1.3 - code_size_max = 110, distance_min = 1m

#CODE_INCLINING_THRESHOLD_DEGREES = 30
CODE_INCLINING_THRESHOLD_DEGREES = 30 * x[11] / 0.5

#CODE_INCLINING_DIRECTION_THRESHOLD_DEGREES = 60
CODE_INCLINING_DIRECTION_THRESHOLD_DEGREES = 60 * x[12] / 0.5

#IS_KERNEL_SHRINKED = x[13] > 0.5
#print('IS_KERNEL_SHRINKED =', IS_KERNEL_SHRINKED) # IS_KERNEL_SHRINKED = False


directions_angle_diff_threshold_cos = np.cos(np.pi * DIRECTIONS_ANGLE_DIFF_THRESHOLD_DEGREES / 180)
code_inclining_threshold_cos = np.cos(np.pi * CODE_INCLINING_THRESHOLD_DEGREES / 180)
code_inclining_direction_threshold_cos = np.cos(np.pi * CODE_INCLINING_DIRECTION_THRESHOLD_DEGREES / 180)
points_gap_size = int(np.round(POINTS_GAP_SIZE_RELATIVE * WIDTH))

kernel_size_square_2 = KERNEL_SIZE_SQUARE ** 2


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

def check_gradient(gradient_cut_unsigned, gradient_threshold):
    gradient_cut_unsigned_sum = np.sum(gradient_cut_unsigned)
    
    x2_mean_tmp = np.sum(X_region_2 * gradient_cut_unsigned) / gradient_cut_unsigned_sum
    #x_mean_tmp = np.sum(X_region * gradient_cut_unsigned) / gradient_cut_unsigned_sum
    y2_mean_tmp = np.sum(Y_region_2 * gradient_cut_unsigned) / gradient_cut_unsigned_sum
    xy_mean_tmp = np.sum(XY_region * gradient_cut_unsigned) / gradient_cut_unsigned_sum
    
    
    
    
    #matrix_tmp = np.asarray([[x2_mean_tmp, xy_mean_tmp], 
    #                         [xy_mean_tmp, y2_mean_tmp]])
    #values_tmp, vectors_tmp = np.linalg.eig(matrix_tmp)
    #max_ind_tmp = np.argmax(values_tmp)
    #main_direction_tmp = vectors_tmp[:, max_ind_tmp]
    
    
    p = x2_mean_tmp
    q = xy_mean_tmp
    r = y2_mean_tmp
    
    # find  eigenvectors:
    # M*v = lambda*v
    # det(M - lambda*I) = 0:
    # (p-lambda*)*(r-lambda*) - q**2 = 0
    # lambda**2 + (-p - r)*lambda + p*r - q**2 = 0
    # solve quadratic polynomial:
    # Discriminant = (p+r)**2 - 4*(p*r-q**2)
    # Discriminant = p**2+2*p*r+r**2 - 4*p*r+4*q**2
    # Discriminant = p**2-2*p*r+r**2 + 4*q**2
    Discriminant = (p-r)**2 + 4*q**2
    sqrt_Discriminant = np.sqrt(Discriminant)
    
    lambda_1 = (p + r + sqrt_Discriminant) / 2
    lambda_2 = (p + r - sqrt_Discriminant) / 2
    
    lambda_t = max(lambda_1, lambda_2)
    
    # solve underdetermeneted system M_t*[a, b] = lambda_t*[a, b], where:
    #M_t = [[p - lambda_t,    q   ],
    #      [q               , r - lambda_t]]
    a_1 = -q
    b_1 = p - lambda_t
    # because in this case (p - lambda_t) * a + q * b = 0
    ab_length2_1 = a_1**2 + b_1**2
    
    
    #a_2 = -(r - lambda_t)
    #b_2 = q
    ## because in this case q * a + q * (r - lambda_t) = 0
    #ab_length2_2 = a_2**2 + b_2**2
    ##print('   *     ', ab_length2_1, ab_length2_2)
    #if ab_length2_1 > ab_length2_2: # choose biggist to avoid devide by zero
    #    ab_length2 = ab_length2_1
    #    a = a_1
    #    b = b_1
    #else:
    #    ab_length2 = ab_length2_2
    #    a = a_2
    #    b = b_2
    #ab_length = np.sqrt(ab_length2)
    #a_normalized = a / ab_length
    #b_normalized = b / ab_length
    
    
    ab_length = np.sqrt(ab_length2_1)
    a_normalized = a_1 / ab_length
    b_normalized = b_1 / ab_length
    
    
    main_direction_tmp = np.asarray([a_normalized, b_normalized])
    
    
    
    
    
    
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
            #print("std_tmp2 =", std_tmp2)
            #print("flatness_tmp6 =", flatness_tmp6)
        
    return is_gradient_ok, main_direction_tmp


def vector_down(direction_tmp):
    if direction_tmp[1] < 0.0:
        direction_tmp = -direction_tmp
    return direction_tmp

def detect_codes(frame):
    x_detected_all = []
    y_detected_all = []
    code_detected_all = []
    code_size_all = []
    x_detected_pairs_all = []
    y_detected_pairs_all = []
    
      
            
    frame = frame[y1_strip: y2_strip, :]
    
        
    frame_orig = np.copy(frame)
    
    frame_orig_gray = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    image_filtered = cv2.filter2D(image_gray, cv2.CV_32F, kernel)
    image_filtered_abs = np.abs(image_filtered)
    retval, image_thresholded = cv2.threshold(image_filtered_abs, FILTERED_THRESHOLD, 255, cv2.THRESH_BINARY)
    image_thresholded = image_thresholded.astype(np.uint8)
    
    
    gradient_x = cv2.filter2D(image_gray, cv2.CV_32F, kernel_gradient_x)
    gradient_y = cv2.filter2D(image_gray, cv2.CV_32F, kernel_gradient_y)
    
    #n_labels, labels = cv2.connectedComponents(image_thresholded, connectivity=4)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_thresholded, connectivity=8)
    
    x_tmp = 0
    points = np.zeros((0, 2), np.int64)
    points_signs = np.zeros(0, np.int64)
    directions = np.zeros((0, 2), np.float32)
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
            
            
            #print(' ')
            #print(' ')
            #print('x')
            gradient_x_cut_unsigned = np.copy(gradient_x_cut)
            gradient_x_cut_unsigned[0:REGION_SIZE,:] = sign * gradient_x_cut_unsigned[0:REGION_SIZE,:]
            gradient_x_cut_unsigned[-REGION_SIZE:,:] = -sign * gradient_x_cut_unsigned[-REGION_SIZE:,:]
            gradient_x_cut_unsigned[gradient_x_cut_unsigned < 0.0] = 0.0
            is_x_gradient_ok, direction_x = check_gradient(gradient_x_cut_unsigned, gradient_threshold)
            #print(' ')
            #print('y')
            gradient_y_cut_unsigned = np.copy(gradient_y_cut)
            gradient_y_cut_unsigned[:, 0:REGION_SIZE] = sign * gradient_y_cut_unsigned[:, 0:REGION_SIZE]
            gradient_y_cut_unsigned[:, -REGION_SIZE:] = -sign * gradient_y_cut_unsigned[:, -REGION_SIZE:]
            gradient_y_cut_unsigned[gradient_y_cut_unsigned < 0.0] = 0.0
            is_y_gradient_ok, direction_y = check_gradient(gradient_y_cut_unsigned, gradient_threshold)
            
            
            if is_x_gradient_ok and is_y_gradient_ok:
                #print('direction_x =', direction_x)
                #frame = cv2.circle(frame, (max_x, max_y), 2, (0, 0, 255), -1)
                
                
                points_tmp = np.zeros((1, 2), np.int64)
                points_tmp[0, 0] = max_x
                points_tmp[0, 1] = max_y
                points = np.concatenate((points, points_tmp), axis=0)
                points_signs = np.append(points_signs, sign)
                
                direction_x = vector_down(direction_x)
                directions_tmp = np.zeros((1, 2), np.float32)
                directions_tmp[0, :] = direction_x
                directions = np.concatenate((directions, directions_tmp), axis=0)
            
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
                
    
    sort_ind_tmp = np.argsort(points[:, 0])
    points = points[sort_ind_tmp, :] 
    #diffs_x_tmp = np.diff(points[:, 0])
    points_signs = points_signs[sort_ind_tmp]
    
    #points_gap_size
    #valid_pairs = np.ones((points.shape[0], points.shape[0]), np.bool)
    valid_points = np.ones(points.shape[0], np.bool)
    valid_pairs = np.zeros((points.shape[0], points.shape[0]), np.bool)
    pairs_directions = np.zeros((points.shape[0], points.shape[0], 2), np.float32)
    points_indexes = np.arange(points.shape[0])
    for point_count_1 in range(0, points.shape[0] - 1):
        point_1 = points[point_count_1, :]
        direction_1 = directions[point_count_1, :]
        for point_count_2 in range(point_count_1 + 1, points.shape[0]):
            point_2 = points[point_count_2, :]
            direction_2 = directions[point_count_2, :]
            
            points_diff_tmp = point_2 - point_1
            distance_tmp = np.sqrt(np.sum(points_diff_tmp**2))
            if distance_tmp <= points_gap_size:
                directions_angle_diff_cos = np.sum(direction_1 * direction_2)
                if np.abs(directions_angle_diff_cos) >= directions_angle_diff_threshold_cos:
                    #print(" ")
                    #print("distance_tmp =", distance_tmp)
                    #print("direction_1 =", direction_1)
                    #print("direction_2 =", direction_2)
                    points_diff_tmp_normalized = points_diff_tmp / distance_tmp
                    if points_diff_tmp_normalized[0] >= code_inclining_threshold_cos:
                        #if points.shape[0] == 6:
                        #    print("180.0 * np.arccos(directions_angle_diff_cos) / np.pi =", 180.0 * np.arccos(directions_angle_diff_cos) / np.pi)
                        
                        valid_pairs[point_count_1, point_count_2] = True
                        
                        #direction_1 = vector_down(direction_1)
                        #direction_2 = vector_down(direction_2)
                        direction_mean = (direction_1 + direction_2) / 2
                        direction_mean = direction_mean / np.sqrt(np.sum(direction_mean**2))
                        pairs_directions[point_count_1, point_count_2, 0] = direction_mean[0]
                        pairs_directions[point_count_1, point_count_2, 1] = direction_mean[1]
                        perpendicular_tmp = np.asarray([-points_diff_tmp_normalized[1], points_diff_tmp_normalized[0]])
                        #a*x + b*y + c = 0
                        # c = -(a*x + b*y)
                        c_tmp = -(perpendicular_tmp[0] * point_1[0] + perpendicular_tmp[1] * point_1[1])
                        perpendicular_distance_tmp = np.abs(perpendicular_tmp[0] * points[:, 0] + perpendicular_tmp[1] * points[:, 1] + c_tmp)
                        longitudial_distance_tmp = points_diff_tmp_normalized[0] * (points[:, 0] - point_1[0]) + points_diff_tmp_normalized[1] * (points[:, 1] - point_1[1])
                        longitudial_distance_normailized_tmp = longitudial_distance_tmp / distance_tmp
                        #print('perpendicular_distance_tmp =', perpendicular_distance_tmp) # ~2
                        #print('longitudial_distance_normailized_tmp =', longitudial_distance_normailized_tmp)
                        
                        condition_indexes_tmp = np.logical_and(points_indexes != point_count_1, points_indexes != point_count_2)
                        condition_longitudial_tmp = np.logical_and(0.0 <= longitudial_distance_normailized_tmp, longitudial_distance_normailized_tmp <= 1.0)
                        condition_distancies_tmp = np.logical_and(condition_longitudial_tmp, perpendicular_distance_tmp <= PERPENDICULAR_DISTANCE_THRESHOLD)
                        condition_tmp = np.logical_and(condition_indexes_tmp, condition_distancies_tmp)
                        
                        valid_points[condition_tmp] = False
                        
            
            #print(" ")
            #print("valid_points =", valid_points)
    
    
    # directions must be close to verticle:
    for point_count in range(0, points.shape[0]):
        direction_tmp = directions[point_count, :]
        if direction_tmp[1] >= code_inclining_direction_threshold_cos:
            pass
        else:
            valid_points[point_count] = False
            #pass
        
    
    #for point_count in range(0, points.shape[0]):
    #    point = points[point_count, :]
    #    if valid_points[point_count]:
    #        color_tmp = (0, 0, 255)
    #    else:
    #        color_tmp = (0,255, 0)
    #    cv2.circle(frame, (int(point[0]), int(point[1])), 2, color_tmp, -1)
    
    
    #x_tmp7 = 10
    #y_tmp7 = 10
    for point_count_1 in range(0, points.shape[0] - 1):
        point_1 = points[point_count_1, :]
        point_1_sign = points_signs[point_count_1]
        #if point_1_sign > 0:
        #    color_tmp1 = (255, 0, 255)
        #else:
        #    color_tmp1 = (255, 255, 0)
        for point_count_2 in range(point_count_1 + 1, points.shape[0]):
            point_2 = points[point_count_2, :]
            point_2_sign = points_signs[point_count_2]
            #if point_2_sign > 0:
            #    color_tmp2 = (255, 0, 255)
            #else:
            #    color_tmp2 = (255, 255, 0)
            if valid_pairs[point_count_1, point_count_2] and \
            valid_points[point_count_1] and valid_points[point_count_2]:
                
                #cv2.circle(frame, (point_1[0], point_1[1]), 4, (0, 255, 255), -1)
                #cv2.circle(frame, (point_2[0], point_2[1]), 2, (0, 255, 255), -1)
                
                #cv2.circle(frame, (point_1[0], point_1[1]), 4, color_tmp1, -1)
                #cv2.circle(frame, (point_2[0], point_2[1]), 4, color_tmp2, -1)
                
                #direction_mean = pairs_directions[point_count_1, point_count_2, :]
                #size_tmp = 10.0
                #direction_mean_sized = size_tmp * direction_mean
                #direction_mean_sized_rounded = np.round(direction_mean_sized).astype(np.int64)
                #cv2.line(img, pt1, pt2, color, th)
                #cv2.line(frame, (point_1[0], point_1[1]), (point_1[0] + direction_mean_sized_rounded[0], point_1[1] + direction_mean_sized_rounded[1]),(0, 255, 255), 1)
                #cv2.line(frame, (point_1[0], point_1[1]), (point_1[0] - direction_mean_sized_rounded[0], point_1[1] - direction_mean_sized_rounded[1]),(0, 255, 255), 1)
                #cv2.line(frame, (point_2[0], point_2[1]), (point_2[0] + direction_mean_sized_rounded[0], point_2[1] + direction_mean_sized_rounded[1]),(0, 255, 255), 1)
                #cv2.line(frame, (point_2[0], point_2[1]), (point_2[0] - direction_mean_sized_rounded[0], point_2[1] - direction_mean_sized_rounded[1]),(0, 255, 255), 1)
                
                
                
                points_diff_tmp = point_2 - point_1
                distance_tmp = np.sqrt(np.sum(points_diff_tmp**2))
                code_size_tmp = distance_tmp
                points_diff_tmp_normalized = points_diff_tmp / distance_tmp
                
                affine_trans_rotation_matrix = np.zeros((2, 3), np.float32)
                
                sign_tmp2 = 1.0
                scale_tmp2 = CODE_CUTTED_UNDISTORTED_SHAPE[0] / distance_tmp
                affine_trans_rotation_matrix[0, 0] = scale_tmp2 * points_diff_tmp_normalized[0]
                affine_trans_rotation_matrix[0, 1] = scale_tmp2 *sign_tmp2 * points_diff_tmp_normalized[1]
                affine_trans_rotation_matrix[1, 0] = - scale_tmp2 * sign_tmp2 * points_diff_tmp_normalized[1]
                affine_trans_rotation_matrix[1, 1] = scale_tmp2 * points_diff_tmp_normalized[0]
                
                point_middle_tmp = (point_2 + point_1) / 2
                affine_trans_rotation_matrix[0, 2] = - affine_trans_rotation_matrix[0, 0] * point_middle_tmp[0] - affine_trans_rotation_matrix[0, 1] * point_middle_tmp[1] + CODE_CUTTED_UNDISTORTED_SHAPE[0] / 2
                affine_trans_rotation_matrix[1, 2] = - affine_trans_rotation_matrix[1, 0] * point_middle_tmp[0] - affine_trans_rotation_matrix[1, 1] * point_middle_tmp[1] + CODE_CUTTED_UNDISTORTED_SHAPE[1] / 2
                
                
                # cv2.warpAffine(src, M, dsize[, dst[, flags[, borderMode[, borderValue]]]]) → dst
                image_rotated_tmp00 = cv2.warpAffine(frame_orig_gray, affine_trans_rotation_matrix, (CODE_CUTTED_UNDISTORTED_SHAPE[0], CODE_CUTTED_UNDISTORTED_SHAPE[1]))
                image_rotated_tmp0 = cv2.resize(image_rotated_tmp00, (CODE_SHAPE[1] - 2, 2))
                #resize_tmp3 = 10
                #image_rotated_tmp = cv2.resize(image_rotated_tmp0, (image_rotated_tmp0.shape[1] * resize_tmp3, image_rotated_tmp0.shape[0] * resize_tmp3), interpolation=cv2.INTER_NEAREST) 
                
                #x_tmp7_2 = x_tmp7 + image_rotated_tmp.shape[1]
                #if x_tmp7_2 < frame.shape[1]:
                #    frame[y_tmp7: y_tmp7 + image_rotated_tmp.shape[0], x_tmp7: x_tmp7 + image_rotated_tmp.shape[1], 0] = image_rotated_tmp
                #    frame[y_tmp7: y_tmp7 + image_rotated_tmp.shape[0], x_tmp7: x_tmp7 + image_rotated_tmp.shape[1], 1] = image_rotated_tmp
                #    frame[y_tmp7: y_tmp7 + image_rotated_tmp.shape[0], x_tmp7: x_tmp7 + image_rotated_tmp.shape[1], 2] = image_rotated_tmp
                #    x_tmp7 = x_tmp7_2 + 10
                
                
                if IS_UPSIDEDOWN:
                    image_rotated_tmp0 = cv2.rotate(image_rotated_tmp0, cv2.ROTATE_180)
                    point_1_sign, point_2_sign = point_2_sign, point_1_sign
                
                black_level_sum = 0.0
                black_level_n = 0
                
                white_level_sum = 0.0
                white_level_n = 0
                
                if point_1_sign > 0:
                    white_level_sum += image_rotated_tmp0[0, 0]
                    white_level_n += 1
                    
                    black_level_sum += image_rotated_tmp0[1, 0]
                    black_level_n += 1
                else:
                    black_level_sum += image_rotated_tmp0[0, 0]
                    black_level_n += 1
                    
                    white_level_sum += image_rotated_tmp0[1, 0]
                    white_level_n += 1
                
                if point_2_sign > 0:
                    black_level_sum += image_rotated_tmp0[0, -1]
                    black_level_n += 1
                    
                    white_level_sum += image_rotated_tmp0[1, -1]
                    white_level_n += 1
                else:
                    
                    white_level_sum += image_rotated_tmp0[0, -1]
                    white_level_n += 1
                    
                    black_level_sum += image_rotated_tmp0[1, -1]
                    black_level_n += 1
                
                black_level_mean = black_level_sum / black_level_n
                white_level_mean = white_level_sum / white_level_n
                
                threshold_black_white = (black_level_mean + white_level_mean) / 2
                
                code_binary_0 = np.zeros(code_binary_length, np.bool)
                code_binary_0[0] = point_1_sign > 0
                code_binary_0[1] = point_2_sign > 0
                code_binary_0[2: 2 + CODE_SHAPE[1] - 4] = image_rotated_tmp0[0, 1:-1] < threshold_black_white
                code_binary_0[2 + CODE_SHAPE[1] - 4:] = image_rotated_tmp0[1, 1:-1] < threshold_black_white
                
                
                check_1 = np.sum(code_binary_0[2: 2 + CODE_SHAPE[1] - 4]) % 2 == 0
                check_2 = np.sum(code_binary_0[2 + CODE_SHAPE[1] - 4:]) % 2 == 0
                check_ok = (code_binary_0[0] == check_1) and (code_binary_0[1] == check_2)
                code_binary = code_binary_0[2:]
                
                    
                
                signal_strengh = np.mean(np.abs(image_rotated_tmp0[:, 1:-1] - threshold_black_white))
                
                if signal_strengh > SIGNAL_STRENGTH_THRESHOLD and check_ok:
                
                    code = np.sum(code_binary * code_binary_bases)
                    
                    #cv2.putText(frame, str(code), (int(point_middle_tmp[0]), int(point_middle_tmp[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    #print('code =', code)
                    
                    x_detected_all.append(point_middle_tmp[0])
                    y_detected_all.append(point_middle_tmp[1])
                    code_detected_all.append(code)
                    code_size_all.append(code_size_tmp)
                    x_detected_pairs_all.append([point_1[0], point_2[0]])
                    y_detected_pairs_all.append([point_1[1], point_2[1]])
    
    
    x_detected_all = np.asarray(x_detected_all)
    y_detected_all = np.asarray(y_detected_all)
    code_detected_all = np.asarray(code_detected_all)
    code_size_all = np.asarray(code_size_all)
    x_detected_pairs_all = np.asarray(x_detected_pairs_all)
    y_detected_pairs_all = np.asarray(y_detected_pairs_all)
    
    
    return x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all