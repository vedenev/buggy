from code_detector_2020_08_24 import detect_codes
import cv2
import numpy as np

PATH = './photos_for_localization_test/028.png'



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

N_SOLVE_ITERATIONS = 10

X_TO_ANGLE_X_LIMITS = [90, 1270]
X_TO_ANGLE_TAILOR_COEFFS = \
    [-0.002227464637962487, 0.3212660622990047, -0.003478867250403373, 0.1909035743393775, -0.0044976791528199165, 0.12648175837018866, -0.005129167400927317]
X_TO_ANGLE_FX = 617.8249050804442
X_TO_ANGLE_CX = 673.0536941293645



def select_codes_in_limits(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all):
    
    
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

frame = cv2.imread(PATH)

x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all = detect_codes(frame)

x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all = \
                select_codes_in_limits(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all)

if x_detected_all.size >= 2: # must be at least two codes
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
    # F_i = 1 - dot_product = 1 - ((x_i - x) / sqrt((x_i - x)**2 + (y_i - y)**2) + (y_i - y) / sqrt((x_i - x)**2 + (y_i - y)**2)
    # F(X) = 0
    # F(X + dX) = 0
    # F(X) + J*dX = 0
    # J*dX = -F(X)
    # X_i+1 = X_i + learning_rate * dX
    # grid search:
    #
    #cs1 = np.linspace(xy_initial[0] - 300.0, xy_initial[0] + 300.0, 201)
    #cs2 = np.linspace(xy_initial[1] - 300.0, xy_initial[1] + 300.0, 201)
    #cs3 = np.linspace(angle_initial - np.pi / 6, angle_initial + np.pi / 6, 101)
    #
    #loss_min = -0.99999976
    #solution_min = [[-6.6800000e+02]
    # [ 1.1991106e+03]
    # [-9.8544323e-01]]
    #dot_products_tmp2_min  = [[0.99999976]
    # [1.        ]
    # [0.99999994]
    # [0.9999999 ]]
    #
    #cs1 = np.linspace(-668 - 100.0, -668 + 100.0, 201)
    #cs2 = np.linspace(1200 - 100.0, 1200 + 100.0, 201)
    #cs3 = np.linspace(-0.985 - np.pi / 12, -0.985 + np.pi / 12, 101)
    #
    #loss_min = -0.99999994
    #solution_min = [[-6.73000e+02]
    # [ 1.18900e+03]
    # [-9.79764e-01]]
    #dot_products_tmp2_min  = [[1.        ]
    # [1.        ]
    # [0.99999994]
    # [1.        ]]
    #
    cs1 = np.linspace(-673 - 12.5, -673 + 12.5, 51)
    cs2 = np.linspace(1189 - 12.5, 1189 + 12.5, 51)
    cs3 = np.linspace(-0.9797 - 0.05, -0.9797 + 0.05, 26)

    cs1, cs2, cs3 = np.meshgrid(cs1, cs2, cs3)
    cs1 = cs1.flatten()
    cs2 = cs2.flatten()
    cs3 = cs3.flatten()
    loss_min = 1.0e6
    solution_min = np.zeros((3, 1), np.float32)
    dot_products_tmp2_min = np.zeros((4, 1), np.float32)
    for grid_index in range(cs1.size):
        current_solution = np.zeros((3, 1), np.float32)
        dot_products_tmp2 = np.zeros((4, 1), np.float32)
        current_solution[0, 0] = cs1[grid_index]
        current_solution[1, 0] = cs2[grid_index]
        current_solution[2, 0] = cs3[grid_index]
        global_index = 0
        loss = 0
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
                loss += ((dx * cos_ray_angle + dy * sin_ray_angle) / distance)
                dot_products_tmp2[global_index] = ((dx * cos_ray_angle + dy * sin_ray_angle) / distance)
                global_index += 1
        
        #loss = np.max(-right_side)
        #loss = -loss
        loss = np.max(-dot_products_tmp2)
        if loss < loss_min:
            loss_min = loss
            solution_min = np.copy(current_solution)
            dot_products_tmp2_min = np.copy(dot_products_tmp2)
    
    current_solution = np.copy(initial_solution)
    #current_solution[0, 0] = -673.0
    #current_solution[1, 0] = 1189.0
    #current_solution[2, 0] = -0.9797
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
                jacobain[global_index, 0] = dy2 * cos_ray_angle / distance3
                jacobain[global_index, 1] = dx2 * sin_ray_angle / distance3
                jacobain[global_index, 2] = - ((dx * (-sin_ray_angle) + dy * cos_ray_angle) / distance)
                
                global_index += 1
        retval, dst	= cv2.solve(jacobain, right_side, None, cv2.DECOMP_SVD)
        
        print(' ')
        print('retval =', retval)
        print('right_side =', right_side)
        print('np.max(-right_side) =', np.max(-right_side))
        print('dot_products =', dot_products)
        print('current_solution =', current_solution)
        
        current_solution = current_solution + 0.5 * dst
    
    print(' ')
    print('loss_min =', loss_min)
    print('solution_min =', solution_min)
    print('dot_products_tmp2_min  =', dot_products_tmp2_min)
        

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
    
    
    
    
    
    

