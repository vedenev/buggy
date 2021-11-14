import numpy as np
import cv2
from code_centres_script import codes_centres
from code_detector_2020_08_24 import y1_strip, y2_strip

ANGLE_PRECISSION = 0.0017 # 600 px <-> 1 radian, angle precission is for 1 px precisioin: 1 / 600
WIDTH_THRESHOLD = 50 * 2
ANGLE_IN_CROSS_THRESHOLD = 0.5 * 53.0 * np.pi / 180.0
COS_ANGLE_IN_CROSS_THRESHOLD = np.cos(ANGLE_IN_CROSS_THRESHOLD)

#N_SOLVE_ITERATIONS = 15
N_SOLVE_ITERATIONS = 30

CODE_SIZE_REAL = 111.0 # mm

MM_IN_CM = 10.0

SIMULARITY_THRESHOD = 0.4
ONE_SOLUSNESS_THRESHOLD = 0.1

codes_centres = np.asarray(codes_centres)
codes_centres[:, 2:4] = MM_IN_CM * codes_centres[:, 2:4]
n_codes = codes_centres.shape[0]

POSITIONS_CODES_REAL = np.zeros((n_codes, 2, 2), np.float32)
#POSITIONS_CODES_REAL[code_no, point_no, x/y]
NOS_CODES_REAL = np.zeros((n_codes), np.int64)

for point_index in range(n_codes):
    code_centre = codes_centres[point_index, :]
    place_no = code_centre[0]
    angle = code_centre[1]
    x = code_centre[2]
    y = code_centre[3]
    
    x11 = - CODE_SIZE_REAL / 2
    y11 = 0.0
    x12 = CODE_SIZE_REAL / 2
    y12 = 0.0
    
    angle_radians = np.pi * angle / 180.0
    sin = np.sin(angle_radians)
    cos = np.cos(angle_radians)
    
    
    
    x21 = x11 * cos + y11 * (-sin)
    y21 = x11 * sin + y11 * cos
    
    x22 = x12 * cos + y12 * (-sin)
    y22 = x12 * sin + y12 * cos
    
    
    
    x31 = x21 + x
    y31 = y21 + y
    
    x32 = x22 + x
    y32 = y22 + y
    
    
    
    POSITIONS_CODES_REAL[point_index, 0, 0] = x31
    POSITIONS_CODES_REAL[point_index, 0, 1] = y31
    POSITIONS_CODES_REAL[point_index, 1, 0] = x32
    POSITIONS_CODES_REAL[point_index, 1, 1] = y32
    
    NOS_CODES_REAL[point_index] = place_no
    #print('NOS_CODES_REAL[point_index] =', NOS_CODES_REAL[point_index])



#PIXELS_TO_ANGLE = np.load("PIXELS_TO_ANGLE.npy")
#PIXELS_TO_ANGLE = PIXELS_TO_ANGLE[y1_strip: y2_strip, :]
    
PIXELS_TO_ANGLE = np.load('PIXELS_TO_ANGLE_2.npy')
PIXELS_TO_ANGLES_TAN_X = np.load('PIXELS_TO_ANGLES_TAN_X_2.npy')
PIXELS_TO_ANGLES_TAN_Y = np.load('PIXELS_TO_ANGLES_TAN_Y_2.npy')

PIXELS_TO_ANGLE = PIXELS_TO_ANGLE[y1_strip: y2_strip, :]
PIXELS_TO_ANGLES_TAN_X = PIXELS_TO_ANGLES_TAN_X[y1_strip: y2_strip, :]
PIXELS_TO_ANGLES_TAN_Y = PIXELS_TO_ANGLES_TAN_Y[y1_strip: y2_strip, :]

#X_TO_ANGLE_X_LIMITS = [90, 1270]
#tmp  = 50
#X_TO_ANGLE_X_LIMITS = [90 + tmp, 1270 - tmp]
X_TO_ANGLE_X_LIMITS = [0, 1280]
X_TO_ANGLE_TAILOR_COEFFS = \
    [-0.002227464637962487, 0.3212660622990047, -0.003478867250403373, 0.1909035743393775, -0.0044976791528199165, 0.12648175837018866, -0.005129167400927317]
X_TO_ANGLE_FX = 617.8249050804442
X_TO_ANGLE_CX = 673.0536941293645


#https://docs.opencv.org/2.4.1/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
mtx = np.array([[617.8249050804442, 0.0, 673.0536941293645], [0.0, 619.3492046143635, 497.9661474464693], [0.0, 0.0, 1.0]])
dist = np.array([[-0.3123562037471547, 0.1018281655721802, 0.00031297833728767365, 0.0007424882126541622, -0.015160446251882953]])

k1 = dist[0, 0]
k2 = dist[0, 1]
p1 = dist[0, 2]
p2 = dist[0, 3]
k3 = dist[0, 4]

fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]


#POSITIONS_CODES_REAL[code_no, point_no, x/y]
# get real codes normals
normals_real_codes = np.zeros((POSITIONS_CODES_REAL.shape[0], 2), np.float32)
tangents_real_codes = POSITIONS_CODES_REAL[:, 1, :] - POSITIONS_CODES_REAL[:, 0, :]
centers_real_codes = (POSITIONS_CODES_REAL[:, 1, :] + POSITIONS_CODES_REAL[:, 0, :]) / 2
tangents_real_codes = tangents_real_codes / np.sqrt(np.sum(tangents_real_codes**2, axis=1, keepdims=True))

normals_real_codes[:, 0] =  -tangents_real_codes[:, 1]
normals_real_codes[:, 1] =  tangents_real_codes[:, 0]


def select_codes_in_limits(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all):
    
    
    if x_detected_all.size > 0:
        condition = (X_TO_ANGLE_X_LIMITS[0] <= x_detected_pairs_all) & (x_detected_pairs_all <= X_TO_ANGLE_X_LIMITS[1])
        condition_codewise = np.all(condition, axis=1)
        
        x_detected_all = x_detected_all[condition_codewise]
        y_detected_all = y_detected_all[condition_codewise]
        code_detected_all = code_detected_all[condition_codewise]
        code_size_all = code_size_all[condition_codewise]
        x_detected_pairs_all = x_detected_pairs_all[condition_codewise, :]
        y_detected_pairs_all = y_detected_pairs_all[condition_codewise, :]
    
    return x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all

def x_to_angle(x):
    xss = (x - X_TO_ANGLE_CX) / X_TO_ANGLE_FX
    xs = np.copy(xss)
    power = np.copy(xss)
    for monom_index in range(len(X_TO_ANGLE_TAILOR_COEFFS)):
        power *= xss
        xs += X_TO_ANGLE_TAILOR_COEFFS[monom_index] * power
    angles = np.arctan(xs)
    return angles

def pixels_to_angle(x, y):
    global PIXELS_TO_ANGLE
    # x_detected_pairs_all[code_no, x1/x2]
    x_flatten = x.flatten()
    y_flatten = y.flatten()
    angles_flatten = PIXELS_TO_ANGLE[y_flatten, x_flatten]
    angles = angles_flatten.reshape(x.shape)
    return angles

def pixels_to_angle_rotation_compensation(x, y):
    global PIXELS_TO_ANGLES_TAN_X
    global PIXELS_TO_ANGLES_TAN_Y
    x_flatten = x.flatten()
    y_flatten = y.flatten()
    tan_x_flatten = PIXELS_TO_ANGLES_TAN_X[y_flatten, x_flatten]
    tan_x = tan_x_flatten.reshape(x.shape)
    tan_y_flatten = PIXELS_TO_ANGLES_TAN_Y[y_flatten, x_flatten]
    mx = np.mean(tan_x_flatten)
    num = np.mean(tan_x_flatten * tan_y_flatten) - mx * np.mean(tan_y_flatten)
    denum = np.mean(tan_x_flatten * tan_x_flatten) - mx * mx
    k = num / denum
    angle_rotation = np.arctan(k)
    tan_x_compensated = tan_x / np.cos(angle_rotation)
    angles = np.arctan(tan_x_compensated)
    return angles

pp = np.asarray([ 7.99509769e-38, -2.42709117e-34, -7.65702590e-31,  5.41010238e-27,
       -1.29585538e-23,  1.74214059e-20, -1.46864380e-17,  8.02857747e-15,
       -2.84100421e-12,  6.31320634e-10, -8.20900190e-08,  5.04597787e-06,
        1.78608889e-03, -1.21520739e+00], np.float64)
def pixels_to_angle_from_measured(x, y):
    global pp
    # y not used
    x_powers = np.ones_like(x).astype(np.float64)
    # pp[-1] = pp[p.size - 1]
    angles = np.full(x.shape, pp[-1], np.float64)
    for index in range(pp.size - 2, -1, -1):
        x_powers = x_powers * x
        angles += pp[index] * x_powers
    
    return angles

def select_existed_codes(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all):
    
    
    if x_detected_all.size > 0:
        
        condition_codewise = np.zeros(x_detected_all.size, np.bool)
        for detected_index in range(code_detected_all.size):
            index_found = np.where(NOS_CODES_REAL == code_detected_all[detected_index])[0]
            if index_found.size == 1:
                condition_codewise[detected_index] = True

        
        
        
        x_detected_all = x_detected_all[condition_codewise]
        y_detected_all = y_detected_all[condition_codewise]
        code_detected_all = code_detected_all[condition_codewise]
        code_size_all = code_size_all[condition_codewise]
        x_detected_pairs_all = x_detected_pairs_all[condition_codewise, :]
        y_detected_pairs_all = y_detected_pairs_all[condition_codewise, :]
    
    return x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all

def select_unique_codes(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all):
    
    if x_detected_all.size >= 2:
        
        #code_detected_all_unique, code_detected_all_unique_counts = np.unique(code_detected_all, return_counts=True)
        #indexes_many = np.nonzero(code_detected_all_unique_counts > 1)[0]
        #code_detected_all_unique_many = code_detected_all_unique[indexes_many]
        #for many_index in range(code_detected_all_unique_many.size):
        #    code_detected_all_unique_many_tmp = code_detected_all_unique_many[many_index]
        #    indexes_of_same = np.nonzero(code_detected_all == code_detected_all_unique_many_tmp)[0]
        #    index_biggest = np.argmax(code_size_all[indexes_of_same])[0]
        
        code_detected_all_unique = np.unique(code_detected_all)
        indexes_to_select = []
        for unique_index in range(code_detected_all_unique.size):
            code_detected_all_unique_tmp = code_detected_all_unique[unique_index]
            indexes = np.nonzero(code_detected_all_unique_tmp == code_detected_all)[0]
            if indexes.size == 1:
                indexes_to_select.append(indexes[0])
            else:
                index_max = np.argmax(code_size_all[indexes])
                indexes_to_select.append(indexes[index_max])
        
        indexes_to_select = np.asarray(indexes_to_select)
        
        x_detected_all = x_detected_all[indexes_to_select]
        y_detected_all = y_detected_all[indexes_to_select]
        code_detected_all = code_detected_all[indexes_to_select]
        code_size_all = code_size_all[indexes_to_select]
        x_detected_pairs_all = x_detected_pairs_all[indexes_to_select, :]
        y_detected_pairs_all = y_detected_pairs_all[indexes_to_select, :]
                
            
            
            
    
    return x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all

def to_positive_angle(angle_):
    if angle_ < 0:
        angle_ += 2 * np.pi
    return angle_

def get_normalized_vector(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    d = np.sqrt(dx**2 + dy**2)
    dxn = dx / d
    dyn = dy / d
    return dxn, dyn

def get_cos_from_3_points(x0, y0, x1, y1, x2, y2):
    dxn1, dyn1 = get_normalized_vector(x0, y0, x1, y1)
    dxn2, dyn2 = get_normalized_vector(x0, y0, x2, y2)
    prod = dxn1 * dxn2 + dyn1 * dyn2
    return prod, dxn1, dyn1, dxn2, dyn2
        
    
def localize_with_circles(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all):
    
    

    
    x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all = \
                            select_codes_in_limits(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all)
    

    
    x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all = \
                            select_existed_codes(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all)
    
    x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all = \
                            select_unique_codes(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all)

    
    is_localized = False
    solution = current_solution = np.full((3, 1), np.NaN, np.float32)
    angles_detected_pairs_all = None
    detected_to_real_indexing = None
    
    

    
    if x_detected_all.size >= 2: # must be at least two codes
        
        
        # x_detected_pairs_all[code_no, x1/x2]
        
        #angles_detected_pairs_all = x_to_angle(x_detected_pairs_all)
        #angles_detected_pairs_all = pixels_to_angle(x_detected_pairs_all, y_detected_pairs_all)
        #angles_detected_pairs_all = pixels_to_angle_rotation_compensation(x_detected_pairs_all, y_detected_pairs_all)
        angles_detected_pairs_all = pixels_to_angle_from_measured(x_detected_pairs_all, y_detected_pairs_all)
        angles_detected_pairs_all =  -angles_detected_pairs_all[:,::-1] # because upsidedown



        ## sort by 1st point, not necessary because it will be resorded by code:
        #sort_indexes = np.argsort(angles_detected_pairs_all[:, 0])
        #angles_detected_pairs_all = angles_detected_pairs_all[sort_indexes, :]
        #code_detected_all = code_detected_all[sort_indexes]
        
        # sort inside pairs in inverse order becuase real and detected orders in pair are oposite:
        for detected_index in range(angles_detected_pairs_all.shape[0]):
            sort_indexes = np.argsort(angles_detected_pairs_all[detected_index, :])[::-1]
            angles_detected_pairs_all[detected_index, :] = angles_detected_pairs_all[detected_index, sort_indexes]
        
        angles_detected_pairs_all = -angles_detected_pairs_all # because camera angles has oposite direction to real angles
        
        angles_detected = (angles_detected_pairs_all[:, 0] + angles_detected_pairs_all[:, 1]) / 2 

        
        #print('code_detected_all =', code_detected_all)
        #print('angles_detected_pairs_all * 180 / np.pi =', angles_detected_pairs_all * 180 / np.pi )
        
        
        angular_sizes_detected_all = angles_detected_pairs_all[:, 1] - angles_detected_pairs_all[:, 0]
        
        
        
        
        # match detected and real indexing by codes:
        
        detected_to_real_indexing = np.zeros(code_detected_all.size, np.int64)
        for detected_index in range(code_detected_all.size):
            index_found = np.where(NOS_CODES_REAL == code_detected_all[detected_index])[0]
            if index_found.size == 1:
                detected_to_real_indexing[detected_index] = index_found
            else:
                print('index_found.size != 1 somthing wrong', index_found.size)
            
        circles = np.zeros((code_detected_all.size, 5), np.float32)
        for code_index in range(code_detected_all.size):
            angles_deltas_tmp_tmp = angular_sizes_detected_all[code_index]
            detected_to_real_indexing_tmp = detected_to_real_indexing[code_index]
            position_code_real = POSITIONS_CODES_REAL[detected_to_real_indexing_tmp, :, :]
            x1 = position_code_real[0, 0]
            y1 = position_code_real[0, 1]
            x2 = position_code_real[1, 0]
            y2 = position_code_real[1, 1]
            
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            
            code_size_tmp = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            
            normals_real_codes_tmp = normals_real_codes[detected_to_real_indexing_tmp, :]
            
            code_size_half_tmp = code_size_tmp / 2
            R_tmp = code_size_half_tmp / np.sin(angles_deltas_tmp_tmp)
            b_tmp = code_size_half_tmp / np.tan(angles_deltas_tmp_tmp)
            
            x_circle = xc + normals_real_codes_tmp[0] * b_tmp
            y_circle = yc + normals_real_codes_tmp[1] * b_tmp
            
            circles[code_index, 0] = x_circle
            circles[code_index, 1] = y_circle
            circles[code_index, 2] = R_tmp
            circles[code_index, 3] = code_size_half_tmp
            circles[code_index, 4] = angles_deltas_tmp_tmp
        
        
        #import matplotlib.pyplot as plt 
        #for circle_index in range(circles.shape[0]):
        #    x_circle = circles[circle_index, 0]
        #    y_circle = circles[circle_index, 1]
        #    R_tmp = circles[circle_index, 2]
        #    a_t = np.linspace(0, 2 * np.pi, 100)
        #    plt.plot(x_circle + R_tmp * np.cos(a_t), y_circle + R_tmp * np.sin(a_t), 'k-')
            
        
        
        P1 = np.zeros(2, np.float32)
        P2 = np.zeros(2, np.float32)
        P3 = np.zeros(2, np.float32)
        
        #print('circles =', circles)
        
        n_circle_pairs = (circles.shape[0] - 1) * circles.shape[0] // 2
        x_cross_all = np.zeros(n_circle_pairs, np.float32)
        y_cross_all = np.zeros(n_circle_pairs, np.float32)
        gamma_all = np.zeros(n_circle_pairs, np.float32)
        is_pair_ok_all = np.ones(n_circle_pairs, np.bool)
        global_index = 0
        for circle_index_1 in range(circles.shape[0] - 1):
            for circle_index_2 in range(circle_index_1 + 1, circles.shape[0]):
                circle_1 = circles[circle_index_1, :]
                index_real_1 = detected_to_real_indexing[circle_index_1]
                real_1 = centers_real_codes[index_real_1, :]
                angle_code_1 = angles_detected[circle_index_1]
                
                circle_2 = circles[circle_index_2, :]
                index_real_2 = detected_to_real_indexing[circle_index_2]
                real_2 = centers_real_codes[index_real_2, :]
                angle_code_2 = angles_detected[circle_index_2]
                
                
                if circle_1[2] < circle_1[2]:
                    circle_2, real_2, index_real_2, angle_code_2, circle_1, real_1, index_real_1, angle_code_1 =\
                    circle_1, real_1, index_real_1, angle_code_1, circle_2, real_2, index_real_2, angle_code_2
                    
                x1 = circle_1[0]
                y1 = circle_1[1]
                r1 = circle_1[2]
                code_size_half_1 = circle_1[3]
                angles_deltas_1 = circle_1[4]
                
                
                x2 = circle_2[0]
                y2 = circle_2[1]
                r2 = circle_2[2]
                code_size_half_2 = circle_2[3]
                angles_deltas_2 = circle_2[4]
                
                

                
                # cross of 2 circles:
                # https://planetcalc.com/8098/
                
                dx = (x2 - x1)
                dy = (y2 - y1)
                
                d = np.sqrt(dx**2 + dy**2)
                
                if d == 0.0 and r1 == r2:
                    #print("something wrong d == 0.0 and r1 == r2, coincendence")
                    is_pair_ok_all[global_index] = False
                if d > (r1 + r2):
                    #print("something wrong d > (r1 + r2), no cross")
                    is_pair_ok_all[global_index] = False
                if d < np.abs(r1 - r2):
                    #print("something wrong d < np.abs(r1 - r2), one inside another")
                    is_pair_ok_all[global_index] = False
                
                #dr = r2 - r1
                #rm = (r1 + r2) / 2
                #simularity = np.sqrt(dx**2 + dy**2 + dr**2) / rm
                #
                #one_solusness = -1.0
                #if is_pair_ok_all[global_index]:
                #    a = (r1**2 - r2**2 + d**2) / (2 * d)
                #    h = np.sqrt(r1**2 - a**2)
                #    one_solusness = h / np.abs(a)
                #    if one_solusness < ONE_SOLUSNESS_THRESHOLD:
                #        is_pair_ok_all[global_index] = False
                #
                ##print("simularity =", simularity)
                #if simularity < SIMULARITY_THRESHOD:
                #    is_pair_ok_all[global_index] = False
                
                if is_pair_ok_all[global_index]:
                    
                    a = (r1**2 - r2**2 + d**2) / (2 * d)
                    h = np.sqrt(r1**2 - a**2)

                    
                    P1[0] = x1
                    P1[1] = y1
                    
                    P2[0] = x2
                    P2[1] = y2
                    
                    P3 = P1 + (a / d) * (P2 - P1)
                    hd = h / d
                    hdx = hd * dx
                    hdy = hd * dy
                    
                    x_cross_1 = P3[0] + hdy
                    y_cross_1 = P3[1] - hdx
                    
                    x_cross_2 = P3[0] - hdy
                    y_cross_2 = P3[1] + hdx
                    
                    #import matplotlib.pyplot as plt
                    #
                    #algles_draw = np.linspace(0, 2 * np.pi, 100)
                    #x_draw = x1 + r1 * np.cos(algles_draw)
                    #y_draw = y1 + r1 * np.sin(algles_draw)
                    #plt.plot(x_draw, y_draw, 'r-')
                    #
                    #x_draw = x2 + r2 * np.cos(algles_draw)
                    #y_draw = y2 + r2 * np.sin(algles_draw)
                    #plt.plot(x_draw, y_draw, 'b-')
                    #
                    #plt.plot([x_cross_1], [y_cross_1], 'kx')
                    #plt.plot([x_cross_2], [y_cross_2], 'k+')
                    #
                    #plt.axis('equal')
                    
                    
                    
                    #import sys
                    #sys.exit()
                    
                    # todo find camera angle using:
                    #x_cross_1
                    #y_cross_1
                    #x_cross_2
                    #y_cross_2
                    #real_1
                    #angle_code_1
                    #real_2
                    #angle_code_2
                    
                    angle_ray_c1_1 = np.arctan2(real_1[1] - y_cross_1, real_1[0] - x_cross_1)
                    #gamma_c1_1 = to_positive_angle(angle_ray_c1_1 - angle_code_1)
                    gamma_c1_1 = angle_ray_c1_1 - angle_code_1
                    cos_gamma_c1_1 = np.cos(gamma_c1_1)
                    sin_gamma_c1_1 = np.sin(gamma_c1_1)
                    angle_ray_c1_2 = np.arctan2(real_2[1] - y_cross_1, real_2[0] - x_cross_1)
                    #gamma_c1_2 = to_positive_angle(angle_ray_c1_2 - angle_code_2)
                    gamma_c1_2 = angle_ray_c1_2 - angle_code_2
                    cos_gamma_c1_2 = np.cos(gamma_c1_2)
                    sin_gamma_c1_2 = np.sin(gamma_c1_2)
                    #consistency_c1 = np.abs(gamma_c1_1 - gamma_c1_2)
                    consistency_c1 = -(cos_gamma_c1_1 * cos_gamma_c1_2 + sin_gamma_c1_1 * sin_gamma_c1_2)
                    
                    angle_ray_c2_1 = np.arctan2(real_1[1] - y_cross_2, real_1[0] - x_cross_2)
                    #gamma_c2_1 = to_positive_angle(angle_ray_c2_1 - angle_code_1)
                    gamma_c2_1 = angle_ray_c2_1 - angle_code_1
                    cos_gamma_c2_1 = np.cos(gamma_c2_1)
                    sin_gamma_c2_1 = np.sin(gamma_c2_1)
                    angle_ray_c2_2 = np.arctan2(real_2[1] - y_cross_2, real_2[0] - x_cross_2)
                    #gamma_c2_2 = to_positive_angle(angle_ray_c2_2 - angle_code_2)
                    gamma_c2_2 = angle_ray_c2_2 - angle_code_2
                    cos_gamma_c2_2 = np.cos(gamma_c2_2)
                    sin_gamma_c2_2 = np.sin(gamma_c2_2)
                    consistency_c2 = - (cos_gamma_c2_1 * cos_gamma_c2_2 + sin_gamma_c2_1 * sin_gamma_c2_2)
                    
                    #print(' ')
                    #print('gamma_c1_1 =', gamma_c1_1 * 180 / np.pi)
                    #print('gamma_c1_2 =', gamma_c1_2 * 180 / np.pi)
                    #print('gamma_c2_1 =', gamma_c2_1 * 180 / np.pi)
                    #print('gamma_c2_2 =', gamma_c2_2 * 180 / np.pi)
                    #
                    #print('angle_ray_c1_1 =', angle_ray_c1_1 * 180 / np.pi)
                    #print('angle_ray_c1_2 =', angle_ray_c1_2 * 180 / np.pi)
                    #print('angle_ray_c2_1 =', angle_ray_c2_1 * 180 / np.pi)
                    #print('angle_ray_c2_2 =', angle_ray_c2_2 * 180 / np.pi)
                    #
                    #print('x_cross_1 =', x_cross_1)
                    #print('y_cross_1 =', y_cross_1)
                    #print('x_cross_2 =', x_cross_2)
                    #print('y_cross_2 =', y_cross_2)
                    #
                    #print('real_1 =', real_1)
                    #print('real_2 =', real_2)
                    #
                    #print('angle_code_1 =', angle_code_1 * 180 / np.pi)
                    #print('angle_code_2 =', angle_code_2 * 180 / np.pi)
                    #print('consistency_c1 =', consistency_c1 * 180 / np.pi)
                    #print('consistency_c2 =', consistency_c2 * 180 / np.pi)
                    
                    if consistency_c1 < consistency_c2:
                        x_cross = x_cross_1
                        y_cross = y_cross_1
                        #gamma = (gamma_c1_1 + gamma_c1_2) / 2
                        x_tmp_gamma = cos_gamma_c1_1 + cos_gamma_c1_2
                        y_tmp_gamma = sin_gamma_c1_1 + sin_gamma_c1_2
                    else:
                        x_cross = x_cross_2
                        y_cross = y_cross_2
                        #gamma = (gamma_c2_1 + gamma_c2_2) / 2
                        x_tmp_gamma = cos_gamma_c2_1 + cos_gamma_c2_2
                        y_tmp_gamma = sin_gamma_c2_1 + sin_gamma_c2_2
                    
                    gamma = np.arctan2(y_tmp_gamma, x_tmp_gamma)
                    
                    cos_phi_1, real_1_xn, real_1_yn, xn_cross_1, yn_cross_1 = get_cos_from_3_points(x1, y1, real_1[0], real_1[1], x_cross, y_cross)
                    width_1 = (code_size_half_1 / (np.sin(angles_deltas_1))**2) * np.abs(np.cos(angles_deltas_1) - cos_phi_1) * ANGLE_PRECISSION
                    
                    cos_phi_2, real_2_xn, real_2_yn, xn_cross_2, yn_cross_2 = get_cos_from_3_points(x2, y2, real_2[0], real_2[1], x_cross, y_cross)
                    width_2 = (code_size_half_2 / (np.sin(angles_deltas_2))**2) * np.abs(np.cos(angles_deltas_2) - cos_phi_2) * ANGLE_PRECISSION
                    
                    cos_in_cross = xn_cross_1 * xn_cross_2 + yn_cross_1 * yn_cross_2
                    
                    #print("width_1 =", width_1)
                    #print("width_2 =", width_2)
                    #print("cos_in_cross =", cos_in_cross)
                    
                    if np.abs(cos_in_cross) >= COS_ANGLE_IN_CROSS_THRESHOLD:
                        is_pair_ok_all[global_index] = False
                    
                    if width_1 >= WIDTH_THRESHOLD:
                        is_pair_ok_all[global_index] = False
                    
                    if width_2 >= WIDTH_THRESHOLD:
                        is_pair_ok_all[global_index] = False
                    
                    x_cross_all[global_index] = x_cross
                    y_cross_all[global_index] = y_cross
                    gamma_all[global_index] = gamma
                
                
                global_index += 1
                
        
        is_pair_ok_indexes_all = np.nonzero(is_pair_ok_all)[0]
        if is_pair_ok_indexes_all.size > 0:
            solution = np.zeros((3, 1), np.float32)
            solution[0] = np.mean(x_cross_all[is_pair_ok_indexes_all])
            solution[1] = np.mean(y_cross_all[is_pair_ok_indexes_all])
            solution[2] = np.mean(gamma_all[is_pair_ok_indexes_all])
                
            is_localized = True
        
    return is_localized, solution, code_detected_all, angles_detected_pairs_all, detected_to_real_indexing
