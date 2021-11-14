from code_detector_2020_08_24 import detect_codes, y1_strip, y2_strip, WIDTH, HEIGHT 
import cv2
import numpy as np
import time
import glob
import os

SELECTED = [17, 20, 24, 28, 31, 36, 40, 43, 47, 55, 58, 62, 66]

DIR = './photos_for_localization_test'
OUTPUT_DIR = './photos_for_localization_test_visualize_tt036'



CODE_SIZE_REAL = 117.0 # mm
CODES_DISTANCE_REAL = 1000.0 # mm
POSITIONS_CODES_REAL = np.zeros((2, 2, 2), np.float32)
#POSITIONS_CODES_REAL[code_no, point_no, x/y]
NOS_CODES_REAL = np.zeros((2), np.int64)

POSITIONS_CODES_REAL[0, 0, 0] = -CODES_DISTANCE_REAL / 2 - CODE_SIZE_REAL / 2
POSITIONS_CODES_REAL[0, 0, 1] = 0.0
POSITIONS_CODES_REAL[0, 1, 0] = -CODES_DISTANCE_REAL / 2 + CODE_SIZE_REAL / 2
POSITIONS_CODES_REAL[0, 1, 1] = 0.0
NOS_CODES_REAL[0] = 25

POSITIONS_CODES_REAL[1, 0, 0] = CODES_DISTANCE_REAL / 2 - CODE_SIZE_REAL / 2
POSITIONS_CODES_REAL[1, 0, 1] = 0.0
POSITIONS_CODES_REAL[1, 1, 0] = CODES_DISTANCE_REAL / 2 + CODE_SIZE_REAL / 2
POSITIONS_CODES_REAL[1, 1, 1] = 0.0
NOS_CODES_REAL[1] = 52

N_SOLVE_ITERATIONS = 15



X_TO_ANGLE_X_LIMITS = [90, 1270]
X_TO_ANGLE_TAILOR_COEFFS = \
    [-0.002227464637962487, 0.3212660622990047, -0.003478867250403373, 0.1909035743393775, -0.0044976791528199165, 0.12648175837018866, -0.005129167400927317]
X_TO_ANGLE_FX = 617.8249050804442
X_TO_ANGLE_CX = 673.0536941293645



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
        
    
#POSITIONS_CODES_REAL[code_no, point_no, x/y]
# get real codes normals
normals_real_codes = np.zeros((POSITIONS_CODES_REAL.shape[0], 2), np.float32)
tangents_real_codes = POSITIONS_CODES_REAL[:, 1, :] - POSITIONS_CODES_REAL[:, 0, :]
centers_real_codes = (POSITIONS_CODES_REAL[:, 1, :] + POSITIONS_CODES_REAL[:, 0, :]) / 2
tangents_real_codes = tangents_real_codes / np.sqrt(np.sum(tangents_real_codes**2, axis=1, keepdims=True))

normals_real_codes[:, 0] =  -tangents_real_codes[:, 1]
normals_real_codes[:, 1] =  tangents_real_codes[:, 0]

time_detect_sum = 0.0
time_detect_n = 0

time_localize_sum = 0.0
time_localize_n = 0

n_frames = 0
        
files = sorted(glob.glob(DIR + '/*.png'))
lc = 0
for index in range(len(files)):
    file_ = files[index]
    file_base = os.path.basename(file_)
    file_base_no_ext = os.path.splitext(file_base)[0]
    file_ok = False
    if len(file_base_no_ext) == 3:
        file_no = int(file_base_no_ext)
        if file_no in SELECTED:
            file_ok = True
    
    if file_ok:
        n_frames += 1
        frame = cv2.imread(file_)
        
        time_1 = time.time()
        x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all = detect_codes(np.copy(frame))
        time_2 = time.time()
        time_detect_sum += time_2 - time_1
        time_detect_n += 1
        
        x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all = \
                        select_codes_in_limits(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all)
        
        is_localized = False
        if x_detected_all.size >= 2: # must be at least two codes
            
            time_1 = time.time()
            
            angles_detected_pairs_all = x_to_angle(x_detected_pairs_all)
            angles_detected_pairs_all =  -angles_detected_pairs_all[:,::-1] # because upsidedown
        
        
            ## sort by 1st point, not necessary because it will be resorded by code:
            #sort_indexes = np.argsort(angles_detected_pairs_all[:, 0])
            #angles_detected_pairs_all = angles_detected_pairs_all[sort_indexes, :]
            #code_detected_all = code_detected_all[sort_indexes]
            
            # sort inside pairs in inverse order becuase real and detected orders in pair are oposite:
            for detected_index in range(angles_detected_pairs_all.shape[0]):
                sort_indexes = np.argsort(angles_detected_pairs_all[detected_index, :])[::-1]
                angles_detected_pairs_all[detected_index, :] = angles_detected_pairs_all[detected_index, sort_indexes]
                
            
            angular_sizes_detected_all = angles_detected_pairs_all[:, 0] - angles_detected_pairs_all[:, 1]
            
            # match detected and real indexing by codes:
            
            detected_to_real_indexing = np.zeros(code_detected_all.size, np.int64)
            for detected_index in range(code_detected_all.size):
                index_found = np.where(NOS_CODES_REAL == code_detected_all[detected_index])[0]
                if index_found.size == 1:
                    detected_to_real_indexing[detected_index] = index_found
                else:
                    print('index_found.size != 1 somthing wrong')
                
            
        
            # initial guiess:
            max_index = np.argmax(angular_sizes_detected_all)
            max_index_real = detected_to_real_indexing[max_index]
            distance_estimate = CODE_SIZE_REAL / angular_sizes_detected_all[max_index]
            xy_initial = centers_real_codes[max_index_real, :] + distance_estimate * normals_real_codes[max_index_real, :]
            directions_for_estimate = centers_real_codes - xy_initial
            directions_for_estimate_normalized = directions_for_estimate / np.sqrt(np.sum(directions_for_estimate **2, axis=1, keepdims=True))
            mean_direction = np.sum(directions_for_estimate_normalized, axis=0)
            #mean_direction = mean_direction / np.sqrt(np.sum(mean_direction**2))
            angle_initial = np.arctan2(mean_direction[1], mean_direction[0])
            
            # iterations:
            right_side = np.zeros(angles_detected_pairs_all.shape[0] * 2, np.float32)
            dot_products = np.zeros(angles_detected_pairs_all.shape[0] * 2, np.float32)
            ray_angle_all = np.zeros(angles_detected_pairs_all.shape[0] * 2, np.float32)
            jacobain = np.zeros((angles_detected_pairs_all.shape[0] * 2, 3), np.float32)
            initial_solution = np.zeros((3, 1), np.float32)
            initial_solution[0, 0] = xy_initial[0]
            initial_solution[1, 0] = xy_initial[1]
            initial_solution[2, 0] = angle_initial
            # F_i = 1 - dot_product = 1 - ((x_i - x) * cos(gamma - alpha) / sqrt((x_i - x)**2 + (y_i - y)**2) + (y_i - y) * sin(gamma - alpha) / sqrt((x_i - x)**2 + (y_i - y)**2)
            # F(X) = 0
            # F(X + dX) = 0
            # F(X) + J*dX = 0
            # J*dX = -F(X)
            # X_i+1 = X_i + learning_rate * dX
            
            #initial_solution[0, 0] = -673.0
            #initial_solution[1, 0] = 1189.0
            #initial_solution[2, 0] = -0.9797
            
        
        
            
            current_solution = np.copy(initial_solution)
            #dst_mean = np.zeros((3, 1), np.float32)
            for iterations_index in range(N_SOLVE_ITERATIONS):
                global_index = 0
                for detected_index in range(angles_detected_pairs_all.shape[0]):
                    index_real = detected_to_real_indexing[detected_index]
                    for in_pair_index in range(2):
                        angle_code = angles_detected_pairs_all[detected_index, in_pair_index]
                        ray_angle = current_solution[2, 0] - angle_code
                        
                        # POSITIONS_CODES_REAL[code_no, point_no, x/y]
                        x_real = POSITIONS_CODES_REAL[index_real, in_pair_index, 0]
                        y_real = POSITIONS_CODES_REAL[index_real, in_pair_index, 1]
                        sin_ray_angle = np.sin(ray_angle)
                        cos_ray_angle = np.cos(ray_angle)
                        dx = (x_real - current_solution[0, 0])
                        dy = (y_real - current_solution[1, 0])
                        dx2 = dx**2
                        dy2 = dy**2
                        distance = np.sqrt(dx2 + dy2)
                        right_side[global_index] = -(1 - ((dx * cos_ray_angle + dy * sin_ray_angle) / distance))
                        dot_products[global_index] = (x_real - current_solution[0, 0]) * cos_ray_angle + (y_real - current_solution[1, 0]) * sin_ray_angle
                        distance3 = distance**3
                        #jacobain[global_index, 0] = dy2 * cos_ray_angle / distance3
                        #jacobain[global_index, 1] = dx2 * sin_ray_angle / distance3
                        jacobain[global_index, 0] = -((dx * (sin_ray_angle * dy + cos_ray_angle * dx) / distance3) - (cos_ray_angle / distance))
                        jacobain[global_index, 1] = -((dy * (sin_ray_angle * dy + cos_ray_angle * dx) / distance3) - (sin_ray_angle / distance))
                        jacobain[global_index, 2] = - ((dx * (-sin_ray_angle) + dy * cos_ray_angle) / distance)
                        
                        global_index += 1
                retval, dst	= cv2.solve(jacobain, right_side, None, cv2.DECOMP_SVD)
                
                #beta = 0.3
                #dst_mean = beta * dst_mean  + (1 - beta) * dst
                
                #print(' ')
                #print('retval =', retval)
                #print('right_side =', right_side)
                #print('np.max(-right_side) =', np.max(-right_side))
                #print('dot_products =', dot_products)
                #print('current_solution =', current_solution)
                
                current_solution = current_solution + 0.7 * dst
                
            is_localized = True
                
            time_2 = time.time()
            
            time_localize_sum += time_2 - time_1
            time_localize_n += 1
        
        # gradient descend:
        #
        # near correct solution:
        #right_side = [-3.9512894e-08 -6.3219254e-08 -1.3028508e-08 -8.8766761e-10]
        #np.max(-right_side) = 6.3219254e-08
        #dot_products = [1194.4984 1211.3252 1629.672  1711.8158]
        #current_solution = [[-6.7300220e+02]
        # [ 1.1889978e+03]
        # [-9.7955722e-01]]
        #
        # from initial guess:
        #retval = True
        #right_side = [-5.5583925e-05 -1.9803223e-05 -8.1461371e-04 -1.1280780e-03]
        #np.max(-right_side) = 0.001128078
        #dot_products = [1270.842  1270.1736 1574.3792 1645.8362]
        #current_solution = [[-4.9224692e+02]
        # [ 1.2691846e+03]
        # [-1.1385413e+00]]
        
        time_iterations = time_2 - time_1
        
        #print('time_iterations =', time_iterations)            
        #print('current_solution =', current_solution)
        #print('right_side =', right_side)
        
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        y_detected_all_frame = y1_strip + y_detected_all 
        x_detected_all_frame = WIDTH - x_detected_all
        y_detected_all_frame = HEIGHT - y_detected_all_frame
        
        y_detected_pairs_all_frame = y1_strip + y_detected_pairs_all
        x_detected_pairs_all_frame = WIDTH - x_detected_pairs_all
        y_detected_pairs_all_frame = HEIGHT - y_detected_pairs_all_frame
        
        for code_index in range(x_detected_all_frame.size):
            cv2.putText(frame, str(code_detected_all[code_index]), (int(x_detected_all_frame[code_index]), int(y_detected_all_frame[code_index])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.circle(frame, (int(x_detected_pairs_all_frame[code_index, 0]), int(y_detected_pairs_all_frame[code_index, 0])), 3, (255, 0, 255), -1) 
            cv2.circle(frame, (int(x_detected_pairs_all_frame[code_index, 1]), int(y_detected_pairs_all_frame[code_index, 1])), 3, (255, 0, 255), -1)
        
        if is_localized:
            y_draw_start = 20
            x_draw_start = 1000
            scale = 0.15
            cv2.rectangle(frame, (x_draw_start - 200, 0), (x_draw_start + 200, 400), (255, 255, 255), -1) 
            for index_real in range(POSITIONS_CODES_REAL.shape[0]):
                    
                # POSITIONS_CODES_REAL[code_no, point_no, x/y]
                x_real_1 = x_draw_start - int(scale * POSITIONS_CODES_REAL[index_real, 0, 0])
                y_real_1 = y_draw_start + int(scale * POSITIONS_CODES_REAL[index_real, 0, 1])
                
                x_real_2 = x_draw_start - int(scale * POSITIONS_CODES_REAL[index_real, 1, 0])
                y_real_2 = y_draw_start + int(scale * POSITIONS_CODES_REAL[index_real, 1, 1])
                
                cv2.line(frame, (x_real_1, y_real_1), (x_real_2, y_real_2), (0, 255, 0), 5)
                
            x_car_found = x_draw_start - int(scale * current_solution[0])
            y_car_found = y_draw_start + int(scale * current_solution[1])
            
            arrow_size = 100.0
            x_car_found_tip = x_car_found - int(scale * arrow_size * np.cos(current_solution[2]))
            y_car_found_tip = y_car_found + int(scale * arrow_size * np.sin(current_solution[2]))
            
            cv2.circle(frame, (x_car_found, y_car_found), 7, (0, 255, 0), -1)
            
            cv2.line(frame, (x_car_found, y_car_found), (x_car_found_tip, y_car_found_tip), (0, 255, 0), 3)
                
                
        y1_strip_frame = HEIGHT - y1_strip    
        y2_strip_frame = HEIGHT - y2_strip    

        cv2.line(frame, (0, y1_strip_frame), (WIDTH, y1_strip_frame), (255, 255, 255), 3)
        cv2.line(frame, (0, y2_strip_frame), (WIDTH, y2_strip_frame), (255, 255, 255), 3)
        
        
        cv2.imwrite(OUTPUT_DIR + '/' + file_base, frame)
    

time_detect_mean = time_detect_sum / time_detect_n
time_localize_mean = time_localize_sum / time_localize_n

print('time_detect_mean =', time_detect_mean)
print('time_detect_n =', time_detect_n)
print('time_localize_mean =', time_localize_mean)
print('time_localize_n =', time_localize_n)
print('n_frames =', n_frames)

#time_detect_mean = 0.1808475164266733
#time_detect_n = 13
#time_localize_mean = 0.019478581168434837
#time_localize_n = 11
#n_frames = 13
#
#time_detect_mean = 0.1745478740105262
#time_detect_n = 13
#time_localize_mean = 0.01935486360029741
#time_localize_n = 11
#n_frames = 13
    
    
    

