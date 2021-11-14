import numpy as np
from code_centres_script import codes_centres
from prepare_vector_field import TARGET_TRAJECTORY
import pickle
import matplotlib.pyplot as plt 

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

not_localized = np.logical_not(is_localized_all)
current_solution_all[not_localized, :] = np.NaN

plt.plot(current_solution_all[:, 0], current_solution_all[:, 1], 'r.-')

code_max = 7
n_correct = 0
n_all = 0
for point_index in range(current_solution_all.shape[0]):
    xt = current_solution_all[point_index, 0]
    yt = current_solution_all[point_index, 1]
    nt = code_detected_all_all[point_index].size
    
    n_correct += np.where( code_detected_all_all[point_index] <= code_max)[0].size
    n_all += nt
    
    #plt.text(xt, yt, str(nt))
    plt.text(xt, yt, str(code_detected_all_all[point_index]))

print(n_correct / n_all) # 0.972891566265060

is_sorted = True
for point_index in range(len(x_detected_all_all)):
    xt = x_detected_all_all[point_index]
    if xt.size > 0:
        if np.any(xt != np.sort(xt)):
            print('not sorted')
            is_sorted = False

print('is_sorted =', is_sorted)

arrow_size = 100.0
for code_index in range(normals_real_codes.shape[0]):
    x1 = centers_real_codes[code_index, 0]
    y1 = centers_real_codes[code_index, 1]
    x2 = x1 + arrow_size * normals_real_codes[code_index, 0]
    y2 = y1 + arrow_size * normals_real_codes[code_index, 1]
    plt.plot([x1, x2], [y1, y2], 'g-')

plt.axis('equal')

mg = 100.0
plt.xlim([np.min(codes_centres[:, 2]) - mg, np.max(codes_centres[:, 2]) + mg])
plt.ylim([np.min(codes_centres[:, 3]) - mg, np.max(codes_centres[:, 3]) + mg])


plt.show()

