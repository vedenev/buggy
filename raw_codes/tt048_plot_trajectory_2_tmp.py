import numpy as np
from code_centres_script import codes_centres
from localize_2020_12_04 import localize
from prepare_vector_field import TARGET_TRAJECTORY
import pickle
import matplotlib.pyplot as plt 

RECALCULATE = True

#PATH_TO_SAVE = 'tt047_saves_2020_12_17'
#PATH_TO_SAVE = 'tt047_saves'
PATH_TO_SAVE = 'tt047_saves_2020_12_21'

is_localized_all = np.load(PATH_TO_SAVE + '/' + 'is_localized_all' + '.npy')
current_solution_all = np.load(PATH_TO_SAVE + '/' + 'current_solution_all' + '.npy')
with open(PATH_TO_SAVE + '/' + 'x_detected_all_all.pickle', 'rb') as handle:
    x_detected_all_all = pickle.load(handle)
with open(PATH_TO_SAVE + '/' + 'y_detected_all_all.pickle', 'rb') as handle:
    y_detected_all_all = pickle.load(handle)
with open(PATH_TO_SAVE + '/' + 'code_detected_all_all.pickle', 'rb') as handle:
    code_detected_all_all = pickle.load(handle)
with open(PATH_TO_SAVE + '/' + 'for_localization.pickle', 'rb') as handle:
    for_localization = pickle.load(handle)

MM_IN_CM = 10.0
CODE_SIZE_REAL = 117.0 # mm

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

plt.figure()
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



if RECALCULATE:
    #plt.close('all')
    is_localized_angle_all = []
    is_localized_all = np.zeros(0, np.bool)
    current_solution_all = np.zeros((0, 3), np.float32)
    for step_index in range(len(for_localization)):
    #for step_index in range(27, 27 + 1):
        for_localization_tmp = for_localization[step_index]
        x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all = for_localization_tmp
        is_localized, current_solution, current_solution_iterations, right_side_iterations = localize(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all)
        if is_localized:
            dist_tmp = np.sqrt((1900 - current_solution[0, 0])**2 +
                (1713 - current_solution[1, 0])**2)
            print('dist_tmp =', dist_tmp)
            if dist_tmp > 2000:
                print('step_index =', step_index)
                #step_index = 63
                #step_index = 79
        
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

plt.show()