#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 08:50:27 2020

@author: vedenev
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

TARGET_DIRECTION = np.asarray([1.0, 0.0])
INITIAL_ANLGE = np.pi / 6
INITIAL_POSITION_X = 0.0
INITIAL_POSITION_Y = 0.0

R_MIN  = 25.0

N_STEPS = 50

dt = 0.01
STEP_TIME = 0.3

X_TURN = 1000.0
FIELD_SIZE = 2000.0
RESOLUTION = 1.0

def flood_fill_distance(map_, start_point_x, start_point_y):
    front = [[start_point_x, start_point_y]]
    size_y = map_.shape[0]
    size_x = map_.shape[1]
    map_distance = np.zeros((size_y, size_x), np.float32)
    map_discovered = np.zeros((size_y, size_x), np.bool)
    map_discovered[start_point_y, start_point_x] = True
    stencil  = [[1, 0],
                [1, 1],
                [0, 1],
                [-1, 1],
                [-1, 0],
                [-1, -1],
                [0, -1],
                [1, -1]]
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
                            front_new.append([point_new_x, point_new_y])
        front = front_new
        if len(front) == 0:
            break
        front_index += 1
        
    return map_discovered, map_distance

def make_zeros_small_vectors_and_normalize(vx, vy, threshold):
    v_norm = np.sqrt(vx**2 + vy**2)
    vx = vx / v_norm
    vx[v_norm < threshold] = 0.0
    vy = vy / v_norm
    vy[v_norm < threshold] = 0.0
    return vx, vy

k_max = 1 / R_MIN

times = np.arange(0, STEP_TIME + dt, dt)

field_size = 2 * int(FIELD_SIZE / RESOLUTION)
field_bool = np.full((field_size, field_size), True, dtype=np.bool)
x_turn = int(X_TURN / RESOLUTION)
filed_center = field_size // 2
x_turn_pixels = filed_center + x_turn
field_bool[filed_center, filed_center : x_turn_pixels] = False
field_bool[filed_center: , x_turn_pixels] = False
field = field_bool.astype(np.uint8)
field = 255 * field
field_dist = cv2.distanceTransform(field, cv2.DIST_L2, 3)

vx = -cv2.Sobel(field_dist, cv2.CV_32F, 1, 0, ksize=3)
vy = -cv2.Sobel(field_dist, cv2.CV_32F, 0, 1, ksize=3)




vx, vy = make_zeros_small_vectors_and_normalize(vx, vy, 4.0)





x = np.arange(field_size)
X, Y = np.meshgrid(x, x)

field_negative = 255 - field
kernel = np.ones((3, 3),np.uint8)
field_negative_increased = cv2.dilate(field_negative, kernel, iterations=1)
map_ = field_negative_increased > 0
start_point_x = x_turn_pixels
start_point_y = field_size - 1
map_discovered, map_distance = flood_fill_distance(map_, start_point_x, start_point_y)


#kernel = np.ones((5,5),np.float32)/25
#map_distance_tmp = cv2.filter2D(map_distance, -1, kernel)
map_distance_tmp = map_distance

vx_path = -cv2.Sobel(map_distance_tmp, cv2.CV_32F, 1, 0, ksize=3)
vy_path = -cv2.Sobel(map_distance_tmp, cv2.CV_32F, 0, 1, ksize=3)

field_negative_bool = field_negative > 0
vx_path = vx_path * field_negative_bool
vy_path = vy_path * field_negative_bool

#kernel = np.ones((5,5),np.float32)/25
#vx_path = cv2.filter2D(vx_path, -1, kernel)
#vy_path = cv2.filter2D(vy_path, -1, kernel)



#vx_path, vy_path = make_zeros_small_vectors_and_normalize(vx_path, vy_path, 4.0)
#field_negative_dist = cv2.distanceTransform(field_negative_increased, cv2.DIST_L2, 3)
#field_negative_dist_tmp = np.clip(field_negative_dist, -100.0, 0.7 * np.max(field_negative_dist))
#max_tmp = np.max(field_negative_dist_tmp)
#field_negative_dist_0 = field_negative_dist_tmp / max_tmp
#field_negative_dist_1 = (max_tmp  - field_negative_dist) / max_tmp
#vx = vx * field_negative_dist_1 + vx_path * field_negative_dist_0
#vy = vy * field_negative_dist_1 + vy_path * field_negative_dist_0

#vx, vy = make_zeros_small_vectors_and_normalize(vx, vy, 0.01)
#v_norm_2 = vx**2 + vy**2
#vx[v_norm_2 == 0.0] = vx_path[v_norm_2 == 0.0]
#vy[v_norm_2 == 0.0] = vy_path[v_norm_2 == 0.0]
#vx, vy = make_zeros_small_vectors_and_normalize(vx, vy, 0.01)

plt.subplot(2, 2, 1)
plt.imshow(field_dist)
plt.colorbar()

plt.subplot(2, 2, 2)
step = 100
plt.quiver(X[::step, ::step], Y[::step, ::step], vx[::step, ::step], -vy[::step, ::step])
plt.gca().invert_yaxis()
plt.axis('equal')
#plt.imshow(v_norm > 4.0)
#plt.colorbar()

plt.subplot(2, 2, 3)
#plt.imshow(map_discovered)

#plt.imshow(map_distance)
#plt.colorbar()

plt.imshow(map_distance.astype(np.int64) % 2 == 0 )



plt.subplot(2, 2, 4)
#plt.imshow(map_distance)
#plt.colorbar()

#plt.quiver(X[::200, ::200], Y[::200, ::200], vx_path[::200, ::200], -vy_path[::200, ::200])
#plt.gca().invert_yaxis()
#plt.axis('equal')

plt.imshow(np.sqrt(vx_path**2 + vy_path**2))
plt.colorbar()


import sys
sys.exit()






state = np.asarray([INITIAL_POSITION_X, INITIAL_POSITION_Y, INITIAL_ANLGE])
state_history = np.copy(state).reshape(1, 3)
state_history_detailed = np.copy(state).reshape(1, 3)


for step_index in range(N_STEPS):
    
    #k = k_max
    #v = 100.0
    if state_history.shape[0] < 3:
        last_gamma = state_history[0, 2]
    else:
        last_gamma = state_history[-3, 2]
    if step_index < 15:
        k = -0.005 * (last_gamma - 0)
    else:
        k = -0.005 * (last_gamma - np.pi/2)
    v = 150.0
    
    dL = v * dt
    dgamma = k * dL
    for time_index in range(times.size):
        state[0] += dL * np.cos(state[2])
        state[1] += dL * np.sin(state[2])
        state[2] += dgamma
        state_history_detailed = np.concatenate((state_history_detailed, state.reshape(1, 3)), axis=0)
    state_history = np.concatenate((state_history, state.reshape(1, 3)), axis=0)

plt.plot(state_history_detailed[:, 0], state_history_detailed[:, 1], 'k-')
plt.plot(state_history[:, 0], state_history[:, 1], 'r.')
plt.axis('equal')