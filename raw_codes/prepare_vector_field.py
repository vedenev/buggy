import numpy as np
import cv2

TARGET_INDEX = 5

# 533 -46 + 79 -36 +16 -322.5 - 19 -28 - 49 -253 = -125.5 - citchen wall to street y-value
# 533 -46 + 79 = 566 # walls max y value
# 323+22+268+55.5+102+19+161.5 = 951 # walls max x value

# 323 + 22 +268 + 55.5 +102 - 19 - (100.5 - 19 - 5.5) / 2 = 713.5
# 323 + 22 +268 + 55.5 +102 + 19 + 100 = 889.5
# 90 cm from M5 243.5  - 90 = 153.5 - old
#TARGET_TRAJECTORY = [[133, 210], 
#                     [254, 210],
#                     [254, 434],
#                     [713.5, 434],
#                     [713.5, 67.5],
#                     [889.5, 67.5]]

TARGET_TRAJECTORY = [[133, 210], 
                     [254, 210],
                     [254, 434],
                     [713.5, 434],
                     [713.5, 67.4],
                     [850.5, 67.4]] # like 20 but (+ 100.5, +62.9)

MM_IN_CM_TMP = 10.0
TARGET_TRAJECTORY = np.asarray(TARGET_TRAJECTORY)
TARGET_TRAJECTORY = MM_IN_CM_TMP * TARGET_TRAJECTORY

TARGET_X = TARGET_TRAJECTORY[TARGET_INDEX, 0]
TARGET_Y = TARGET_TRAJECTORY[TARGET_INDEX, 1]

RESOLUTION = 0.1 # px / mm

FIELD_SIZE_X_MM = 10000.0
FIELD_SIZE_Y_MM = 6000.0

PATH_DILATE_SIZE = 5
resize_multiplyer = 5

IS_DRAW = False

def make_zeros_small_vectors_and_normalize(vx, vy, threshold):
    v_norm = np.sqrt(vx**2 + vy**2)
    vx = vx / v_norm
    vx[v_norm < threshold] = 0.0
    vy = vy / v_norm
    vy[v_norm < threshold] = 0.0
    return vx, vy

def flood_fill_distance_and_direction(map_, start_point_x, start_point_y):
    front = [[start_point_x, start_point_y]]
    size_y = map_.shape[0]
    size_x = map_.shape[1]
    map_distance = np.zeros((size_y, size_x), np.float32)
    vx_path = np.zeros((size_y, size_x), np.float32)
    vy_path = np.zeros((size_y, size_x), np.float32)
    map_discovered = np.zeros((size_y, size_x), np.bool)
    map_discovered[start_point_y, start_point_x] = True
    stencil = [[1, 0],
               [0, 1],
               [-1, 0],
               [0, -1]]
    front_index = 1
    while True: # front steps
        front_new = []
        for point_index in range(len(front)):
            point_x, point_y = front[point_index]
            for stencil_index in range(len(stencil)):
                shift_x, shift_y = stencil[stencil_index]
                point_new_x = point_x + shift_x
                point_new_y = point_y + shift_y
                condition = 0 <= point_new_x and point_new_x < size_x
                condition = condition and 0 <= point_new_y and point_new_y < size_y
                if condition:
                    if map_[point_new_y, point_new_x]:
                        if not map_discovered[point_new_y, point_new_x]:
                            map_discovered[point_new_y, point_new_x] = True
                            map_distance[point_new_y, point_new_x] = front_index
                            vx_path[point_new_y, point_new_x] = -shift_x
                            vy_path[point_new_y, point_new_x] = -shift_y
                            front_new.append([point_new_x, point_new_y])
        front = front_new
        if len(front) == 0:
            break
        front_index += 1
        
    return map_discovered, map_distance, vx_path, vy_path


field_size_x = int(FIELD_SIZE_X_MM * RESOLUTION)
field_size_y = int(FIELD_SIZE_Y_MM * RESOLUTION)
field = np.full((field_size_y, field_size_x), 255, dtype=np.uint8)
target_trajectory_points_to_draw = (TARGET_TRAJECTORY * RESOLUTION).astype(np.int32).reshape(-1, 1, 2)
field = cv2.polylines(field, [target_trajectory_points_to_draw], False, (0, 0, 0), 1) # todo: why not draw?
field_bool = field > 0
#field_uint8 = cv2.line(field_uint8, (0, 0), (300, 500), (0, 0, 0), 1)
#field_uint8 = polyline(field_uint8, target_trajectory_points_to_draw)

#points = 
#for point_index in range(points.shape[0] - 1):
#    point_1 = tuple(points[point_index, :])
#    point_2 = tuple(points[point_index + 1, :])
#    field_uint8 = cv2.line(field_uint8, point_1, point_2, (0, 0, 0), 1)

field_dist = cv2.distanceTransform(field, cv2.DIST_L2, 3)

vx = -cv2.Sobel(field_dist, cv2.CV_32F, 1, 0, ksize=3)
vy = -cv2.Sobel(field_dist, cv2.CV_32F, 0, 1, ksize=3)




vx, vy = make_zeros_small_vectors_and_normalize(vx, vy, 4.0)





y = np.arange(field_size_y)
x = np.arange(field_size_x)
X, Y = np.meshgrid(x, y)

# np.round((X + FIELD_SIZE_HALF_MM) / RESOLUTION) - pixels to mm



field_negative = 255 - field
map_ = field_negative > 0
start_point_x = target_trajectory_points_to_draw[-1, 0, 0]
start_point_y = target_trajectory_points_to_draw[-1, 0, 1]
map_discovered, map_distance, vx_path, vy_path = flood_fill_distance_and_direction(map_, start_point_x, start_point_y)



size_cv2_tmp = (vx_path.shape[1] // resize_multiplyer, vx_path.shape[0] // resize_multiplyer)
kernel_tmp2 = np.ones((resize_multiplyer, resize_multiplyer), np.uint8)
vx_path = cv2.dilate(vx_path, kernel_tmp2) - cv2.dilate(-vx_path, kernel_tmp2)
vx_path = cv2.resize(vx_path, size_cv2_tmp, interpolation=cv2.INTER_NEAREST)
vy_path = cv2.dilate(vy_path, kernel_tmp2) - cv2.dilate(-vy_path, kernel_tmp2)
vy_path = cv2.resize(vy_path, size_cv2_tmp, interpolation=cv2.INTER_NEAREST)




kernel = np.ones((PATH_DILATE_SIZE, PATH_DILATE_SIZE),np.float32)
kernel = kernel / np.sum(kernel)
for iteration_index in range(11):
    vx_path = cv2.filter2D(vx_path, -1, kernel)
    vy_path = cv2.filter2D(vy_path, -1, kernel)


size_cv2_tmp = (vx.shape[1], vx.shape[0])
vx_path = cv2.resize(vx_path, size_cv2_tmp, interpolation=cv2.INTER_NEAREST)
vy_path = cv2.resize(vy_path, size_cv2_tmp, interpolation=cv2.INTER_NEAREST)


#plt.imshow(np.sqrt(vx_path**2 + vy_path**2))
#plt.colorbar()
#plt.show()
#import sys
#sys.exit()



notm_tmp = np.sqrt(vx_path**2 + vy_path**2)

vx_path = vx_path / notm_tmp
vy_path = vy_path / notm_tmp

max_tmp = np.max(notm_tmp)
notm_tmp = notm_tmp / max_tmp
notm_tmp[notm_tmp < 0.01] = 0.0
vx_path[notm_tmp == 0.0] = 0.0
vy_path[notm_tmp == 0.0] = 0.0

mask_tmp_0 = notm_tmp
mask_tmp_1 = 1.0 - notm_tmp        

#mask_tmp_0 = 0.0 + (notm_tmp  > 0.9)
#mask_tmp_1 = 1.0 - (notm_tmp > 0.9)

vx = mask_tmp_1 * vx + mask_tmp_0 * vx_path
vy = mask_tmp_1 * vy + mask_tmp_0 * vy_path
vx, vy = make_zeros_small_vectors_and_normalize(vx, vy, 0.01)

if IS_DRAW:
    import matplotlib.pyplot as plt 
    plt.subplot(2, 2, 1)
    step = 30
    plt.quiver(X[::step, ::step], Y[::step, ::step], vx_path[::step, ::step], -vy_path[::step, ::step])
    plt.gca().invert_yaxis()
    plt.axis('equal')

    plt.subplot(2, 2, 2)
    step = 30
    plt.quiver(X[::step, ::step], Y[::step, ::step], vx[::step, ::step], -vy[::step, ::step])
    plt.plot(TARGET_TRAJECTORY[:, 0] * RESOLUTION, TARGET_TRAJECTORY[:, 1] * RESOLUTION, 'g-')
    plt.gca().invert_yaxis()
    plt.axis('equal')

    plt.subplot(2, 2, 3)
    plt.imshow(mask_tmp_0)

    plt.subplot(2, 2, 4)
    step = 10
    plt.quiver(MM_IN_CM_TMP * X[::step, ::step], MM_IN_CM_TMP * Y[::step, ::step], vx[::step, ::step], vy[::step, ::step])
    plt.plot(MM_IN_CM_TMP * TARGET_TRAJECTORY[:, 0] * RESOLUTION, MM_IN_CM_TMP * TARGET_TRAJECTORY[:, 1] * RESOLUTION, 'g-')
    plt.axis('equal')

    #plt.show()



