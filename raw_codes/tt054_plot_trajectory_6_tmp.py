import numpy as np
from code_centres_script import codes_centres
from localize_with_circles_2020_12_29 import localize_with_circles
from localize_with_circles_2020_12_29_tmp import localize_with_circles_tmp
from prepare_vector_field import TARGET_TRAJECTORY, vx, vy
import pickle
import matplotlib.pyplot as plt 




PATH_TO_SAVE = 'tt053_saves'

is_localized_all = np.load(PATH_TO_SAVE + '/' + 'is_localized_all' + '.npy')
is_localized_by_odometry_all = np.load(PATH_TO_SAVE + '/' + 'is_localized_by_odometry_all' + '.npy')
current_solution_all = np.load(PATH_TO_SAVE + '/' + 'current_solution_all' + '.npy')
counter_right_all = np.load(PATH_TO_SAVE + '/' + 'counter_right_all' + '.npy')
counter_left_all = np.load(PATH_TO_SAVE + '/' + 'counter_left_all' + '.npy')
right_value_all = np.load(PATH_TO_SAVE + '/' + 'right_value_all' + '.npy')
left_value_all = np.load(PATH_TO_SAVE + '/' + 'left_value_all' + '.npy')
with open(PATH_TO_SAVE + '/' + 'x_detected_all_all.pickle', 'rb') as handle:
    x_detected_all_all = pickle.load(handle)
with open(PATH_TO_SAVE + '/' + 'y_detected_all_all.pickle', 'rb') as handle:
    y_detected_all_all = pickle.load(handle)
with open(PATH_TO_SAVE + '/' + 'code_detected_all_all.pickle', 'rb') as handle:
    code_detected_all_all = pickle.load(handle)
with open(PATH_TO_SAVE + '/' + 'for_localization.pickle', 'rb') as handle:
    for_localization = pickle.load(handle)

MM_IN_CM = 10.0
CODE_SIZE_REAL = 111.0 # mm

codes_centres = np.asarray(codes_centres)
codes_centres[:, 2:4] = MM_IN_CM * codes_centres[:, 2:4]

n_codes = codes_centres.shape[0]
positions_codes_real = np.zeros((n_codes, 2, 2), np.float32)
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
    
    
    
    positions_codes_real[point_index, 0, 0] = x31
    positions_codes_real[point_index, 0, 1] = y31
    positions_codes_real[point_index, 1, 0] = x32
    positions_codes_real[point_index, 1, 1] = y32
    

normals_real_codes = np.zeros((positions_codes_real.shape[0], 2), np.float32)
tangents_real_codes = positions_codes_real[:, 1, :] - positions_codes_real[:, 0, :]
centers_real_codes = (positions_codes_real[:, 1, :] + positions_codes_real[:, 0, :]) / 2
tangents_real_codes = tangents_real_codes / np.sqrt(np.sum(tangents_real_codes**2, axis=1, keepdims=True))

normals_real_codes[:, 0] =  -tangents_real_codes[:, 1]
normals_real_codes[:, 1] =  tangents_real_codes[:, 0]

#plt.figure()
plt.plot(TARGET_TRAJECTORY[:, 0], TARGET_TRAJECTORY[:, 1], 'k.-')
for point_index in range(n_codes):
    position_code_real = positions_codes_real[point_index, :, :]
    x1 = position_code_real[0, 0]
    y1 = position_code_real[0, 1]
    x2 = position_code_real[1, 0]
    y2 = position_code_real[1, 1]
    plt.plot([x1], [y1], 'kd')
    plt.plot([x2], [y2], 'ks')
    plt.plot([x1, x2], [y1, y2], 'k-')
    
    palace_no = codes_centres[point_index, 0]
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    shift = 100
    plt.text(x + shift, y + shift, str(int(palace_no)))

RECALCULATE = False
#step_index_to_see = [92]
step_index_to_see = [83]


if RECALCULATE:
    #plt.close('all')
    is_localized_angle_all = []
    is_localized_all = np.zeros(0, np.bool)
    current_solution_all = np.zeros((0, 3), np.float32)
    code_detected_all_all = []
    for step_index in range(len(for_localization)):
    #for step_index in range(0, 0 + 1):
        for_localization_tmp = for_localization[step_index]
        x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all = for_localization_tmp
        #if step_index in [33, 34, 35, 36, 37, 38, 39]:
        #if step_index in [37, 39]:
        if step_index in step_index_to_see:
        #if step_index in [62]:
            print(' ')
            #x_detected_pairs_all[detected_code_no][point_1_of_code/point_2_of_code] = x
            print('step_index =', step_index)
            print('code_detected_all =', code_detected_all)
            print('x_detected_pairs_all =', x_detected_pairs_all)
            is_localized, current_solution, code_detected_all, angles_detected_pairs_all, detected_to_real_indexing = localize_with_circles_tmp(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all)
        else:
            is_localized, current_solution, code_detected_all, angles_detected_pairs_all, detected_to_real_indexing = localize_with_circles(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all)
        code_detected_all_all.append(code_detected_all)
        #if is_localized:
        #    dist_tmp = np.sqrt((1900 - current_solution[0, 0])**2 +
        #        (1713 - current_solution[1, 0])**2)
        #    print('dist_tmp =', dist_tmp)
        #    if dist_tmp > 2000:
        #        print('step_index =', step_index)
        #        #step_index = 63
        #        #step_index = 79
        
        if not is_localized:
            current_solution = np.full((3, 1), np.NaN, np.float32)
        
        if is_localized:
            is_localized_angle_all.append(current_solution[2, 0])
            
            #print('step_index =', step_index) # step_index = 27
            #import sys
            #sys.exit()
            
        
        
            
        
            
        is_localized_all = np.append(is_localized_all, is_localized)
        current_solution_all = np.concatenate((current_solution_all, current_solution.reshape(1, 3)), axis=0)
        
        #print("current_solution_iterations =", current_solution_iterations)
        
        
        
      
        
        ##plt.plot(current_solution_iterations[0, :], current_solution_iterations[1, :], 'g.-')
        ##for iteration_index in range(current_solution_iterations.shape[1]):
        #for iteration_index in range(current_solution_iterations.shape[1] - 1, current_solution_iterations.shape[1]):
        #    #plt.plot(current_solution_iterations[0, iteration_index], current_solution_iterations[1, iteration_index], 'g.')
        #    #plt.text(current_solution_iterations[0, iteration_index], current_solution_iterations[1, iteration_index], str(iteration_index))
        #    
        #    plt.plot(current_solution_iterations[0, iteration_index], current_solution_iterations[1, iteration_index], 'g.')
        #    ar_s = 3000.0
        #    plt.plot([current_solution_iterations[0, iteration_index], current_solution_iterations[0, iteration_index] + ar_s * np.cos(current_solution_iterations[2, iteration_index])], [current_solution_iterations[1, iteration_index], current_solution_iterations[1, iteration_index] + ar_s * np.sin(current_solution_iterations[2, iteration_index])], 'g-')
        
        #plt.axis('equal')
        
        
        #plt.figure()
        #plt.plot(right_side_iterations[:, 0], 'r.-')
        #plt.plot(right_side_iterations[:, 1], 'gx-')
        #plt.plot(right_side_iterations[:, 2], 'bd-')
        #plt.plot(right_side_iterations[:, 3], 'ks-')
        #
        #plt.show()

plt.plot(current_solution_all[:, 0], current_solution_all[:, 1], 'b.-')


is_nan = np.isnan(current_solution_all[:, 0])
is_nan_extended = np.concatenate(([False], is_nan, [False]))
is_nan_diff = np.diff(is_nan_extended.astype(np.int64))
is_nan_starts = np.where(is_nan_diff > 0)[0] - 1
is_nan_ends = np.where(is_nan_diff < 0)[0]
for nan_rages_index in range(is_nan_starts.size):
    starts_tmp = is_nan_starts[nan_rages_index]
    ends_tmp = is_nan_ends[nan_rages_index]
    if 0 <= starts_tmp and ends_tmp < current_solution_all.shape[0]:
        plt.plot([current_solution_all[starts_tmp, 0], current_solution_all[ends_tmp, 0]], [current_solution_all[starts_tmp, 1], current_solution_all[ends_tmp, 1]], 'r.-')
        print('current_solution_all[starts_tmp, :] =', current_solution_all[starts_tmp, :])
        print('current_solution_all[ends_tmp, :] =', current_solution_all[ends_tmp, :])


start_index = 145
end_index = -160
N_holes = 20
r_wheel = 31 # mm
a_wheels = 118 # mm, half-distance between wheels
pi_r_N = np.pi * r_wheel / N_holes
pi_r_N_a = pi_r_N  / a_wheels


for step_index in range(len(for_localization)):
    if RECALCULATE:
        if step_index not in step_index_to_see:
            continue
    code_detected_all_2 = code_detected_all_all[step_index]
    for_localization_tmp = for_localization[step_index]
    x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all = for_localization_tmp
    current_solution = current_solution_all[step_index, :]
    code_detected_all = code_detected_all_all[step_index]
    is_localized_by_odometry = is_localized_by_odometry_all[step_index]
    counter_right = counter_right_all[step_index]
    counter_left = counter_left_all[step_index]
    right_value = right_value_all[step_index]
    left_value = left_value_all[step_index]
    if step_index == start_index:
        counter_right_old = counter_right;
        counter_left_old = counter_left;
        x_old = current_solution[0]
        y_old = current_solution[1]
        gamma_old = current_solution[2]
    
    if start_index <= step_index and step_index <= end_index:
    #if start_index == step_index:
        delta_right = counter_right - counter_right_old
        delta_left = counter_left - counter_left_old
        alpha = pi_r_N_a * (delta_right - delta_left)
        gamma_new = gamma_old + alpha
        amplitude_tmp = (delta_right  + delta_left) * pi_r_N * np.sinc(alpha / (2 * np.pi))
        angle_tmp = gamma_old + alpha / 2
        x_new = x_old + amplitude_tmp * np.cos(angle_tmp);
        y_new = y_old + amplitude_tmp * np.sin(angle_tmp);
        plt.plot(x_new, y_new, 'r+')
        plt.text(x_new, y_new, str(step_index))
        plt.plot([x_new, x_new + ar_s * np.cos(gamma_new)], [y_new, y_new + ar_s * np.sin(gamma_new)], 'r-')
        
        counter_right_old = counter_right
        counter_left_old = counter_left
        x_old = x_new
        y_old = y_new
        gamma_old = gamma_new
    
    ar_s = 100.0
    if not np.isnan(current_solution[0]):
        plt.plot(current_solution[0], current_solution[1], 'g.')
        if is_localized_by_odometry:
            plt.plot(current_solution[0], current_solution[1], 'gx')
        plt.text(current_solution[0], current_solution[1], str(code_detected_all) + ' ' + str(code_detected_all_2) + ' ' + str(step_index) + ' ' + str(counter_right) + ' ' + str(counter_left) + ' ' + str(right_value) + ' ' + str(left_value))
        plt.plot([current_solution[0], current_solution[0] + ar_s * np.cos(current_solution[2])], [current_solution[1], current_solution[1] + ar_s * np.sin(current_solution[2])], 'g-')


##x_old = 5888.84
##y_old = 4912.92
##gamma_old = np.pi / 6
##
##delta_right = 1668 - 1591
##delta_left = 1527 - 1518
#
#x_old = 3807.4
#y_old = 4689.36
#gamma_old = -np.pi / 6
#
#delta_right = 1381 - 1364
#delta_left = 1335 - 1305
#
#for tmp in np.linspace(0, 1, 30):
#    delta_right_tmp = delta_right * tmp
#    delta_left_tmp = delta_left * tmp
#    alpha = pi_r_N_a * (delta_right_tmp - delta_left_tmp)
#    gamma_new = gamma_old + alpha
#    amplitude_tmp = (delta_right_tmp  + delta_left_tmp ) * pi_r_N * np.sinc(alpha / (2 * np.pi))
#    angle_tmp = gamma_old + alpha / 2
#    x_new = x_old + amplitude_tmp * np.cos(angle_tmp);
#    y_new = y_old + amplitude_tmp * np.sin(angle_tmp);
#
#    plt.plot(x_new, y_new, 'c.')

##plt.plot(x_new, y_new, 'c+')

plt.axis('equal')
#plt.xlim([np.min(positions_codes_real[:, :, 0]), np.max(positions_codes_real[:, :, 0])])
#plt.ylim([np.min(positions_codes_real[:, :, 1]), np.max(positions_codes_real[:, :, 1])])
plt.show()

