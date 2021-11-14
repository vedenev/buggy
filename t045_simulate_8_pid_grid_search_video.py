#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 08:50:27 2020

@author: vedenev
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Use Agg backend for canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from decimal import Decimal

INITIAL_ANLGE = -np.pi / 6
INITIAL_POSITION_X = 300.0
INITIAL_POSITION_Y = 1000.0

EDGE_MARGIN = 3

OUTPUT_VIDEO = 'pid_regulator.mp4'
FPS = 10.0

R_MIN  = 25.0

N_STEPS = 150

dt = 0.01
STEP_TIME = 0.3

X_TURN = 1000.0
FIELD_SIZE_HALF_MM = 2000.0 # half field size in mm
RESOLUTION = 1.0 # px / mm ?

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

def make_zeros_small_vectors_and_normalize(vx, vy, threshold):
    v_norm = np.sqrt(vx**2 + vy**2)
    vx = vx / v_norm
    vx[v_norm < threshold] = 0.0
    vy = vy / v_norm
    vy[v_norm < threshold] = 0.0
    return vx, vy

def get_target_direction(last_x, last_y, vx, vy):
    #FIELD_SIZE_HALF_MM = 2000.0
    #RESOLUTION = 1.0
    last_x_pixels = int(np.round((last_x) / RESOLUTION))
    last_y_pixels = int(np.round((last_y) / RESOLUTION))
    condition = EDGE_MARGIN <= last_x_pixels
    condition = condition and last_x_pixels <= vx.shape[1] - 1 - EDGE_MARGIN
    condition = condition and EDGE_MARGIN <= last_y_pixels
    condition = condition and last_y_pixels <= vx.shape[0] - 1 - EDGE_MARGIN
    if condition:
        target_direction_x = vx[last_y_pixels, last_x_pixels]
        target_direction_y = vy[last_y_pixels, last_x_pixels]
    else:
        target_direction_x = None
        target_direction_y = None
    return target_direction_x, target_direction_y, condition

k_max = 1 / R_MIN

times = np.arange(0, STEP_TIME + dt, dt)

field_size = 2 * int(FIELD_SIZE_HALF_MM / RESOLUTION)
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

# np.round((X + FIELD_SIZE_HALF_MM) / RESOLUTION) - pixels to mm

X_pixels = RESOLUTION * X
Y_pixels = RESOLUTION * Y

field_negative = 255 - field
map_ = field_negative > 0
start_point_x = x_turn_pixels
start_point_y = field_size - 1
map_discovered, map_distance, vx_path, vy_path = flood_fill_distance_and_direction(map_, start_point_x, start_point_y)

resize_multiplyer = 10
size_cv2_tmp = (vx_path.shape[1] // resize_multiplyer, vx_path.shape[0] // resize_multiplyer)
vx_path = cv2.resize(vx_path, size_cv2_tmp, interpolation=cv2.INTER_NEAREST)
vy_path = cv2.resize(vy_path, size_cv2_tmp, interpolation=cv2.INTER_NEAREST)

kernel = np.ones((15, 15),np.float32)
kernel = kernel / np.sum(kernel)
for iteration_index in range(11):
    vx_path = cv2.filter2D(vx_path, -1, kernel)
    vy_path = cv2.filter2D(vy_path, -1, kernel)
    
size_cv2_tmp = (vx.shape[1], vx.shape[0])
vx_path = cv2.resize(vx_path, size_cv2_tmp, interpolation=cv2.INTER_NEAREST)
vy_path = cv2.resize(vy_path, size_cv2_tmp, interpolation=cv2.INTER_NEAREST)

#plt.imshow(np.sqrt(vx_path**2 + vy_path**2))
#plt.colorbar()

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


#plt.subplot(2, 2, 1)
#step = 50
#plt.quiver(X[::step, ::step], Y[::step, ::step], vx_path[::step, ::step], -vy_path[::step, ::step])
#plt.gca().invert_yaxis()
#plt.axis('equal')

#plt.subplot(2, 2, 2)
#step = 50
#plt.quiver(X[::step, ::step], Y[::step, ::step], vx[::step, ::step], -vy[::step, ::step])
#plt.gca().invert_yaxis()
#plt.axis('equal')

#plt.subplot(2, 2, 3)
#plt.imshow(mask_tmp_0 > 0.5)


#loopback_stength_all = 10 ** np.logspace(np.log10(0.0001), 0.1, 200)
loopback_stength_all = 10 ** np.linspace(np.log10(0.0001), np.log10(0.1), 100)

#k_i = 0.00001
#k_d = 0.0

#k_i = 0.0
#k_d = 0.001

#N = 5
#k_p = np.linspace(0.001, 0.003, N)
#k_i = np.linspace(0.000003, 0.00003, N)
#k_d = np.linspace(0.0003, 0.003, N)

N = 15
k_p = np.linspace(0.0005, 0.005, N)
k_i = np.linspace(0.000001, 0.00005, N)
k_d = np.linspace(0.0001, 0.005, N)
#residual_min = 41.74995885178227
#k_p_min = 0.0017857142857142859
#k_i_min = 4.6500000000000005e-05
#k_d_min = 0.0001

k_p, k_i, k_d = np.meshgrid(k_p, k_i, k_d)

k_p = k_p.flatten()
k_i = k_i.flatten()
k_d = k_d.flatten()

residual_min = 1.0e30
k_p_min = None
k_i_min = None
k_d_min = None

fig = plt.figure()

for frame_index in range(k_p.size):
    
    plt.cla()
    
    k_p_tmp = k_p[frame_index]
    k_i_tmp = k_i[frame_index]
    k_d_tmp = k_d[frame_index]
    
    state = np.asarray([INITIAL_POSITION_X, INITIAL_POSITION_Y, INITIAL_ANLGE])
    state_history = np.copy(state).reshape(1, 3)
    state_history_detailed = np.copy(state).reshape(1, 3)
    
    angle_delta_old = 0.0
    angle_delta_old_2 = 0.0
    U_old = 0.0
    U_old_2 = 0.0
    
    residual = 0.0
    
    for step_index in range(N_STEPS):
        
        #k = k_max
        #v = 100.0
        if state_history.shape[0] < 3:
            last_gamma = state_history[0, 2]
            last_x = state_history[0, 0]
            last_y = state_history[0, 1]
        else:
            last_gamma = state_history[-3, 2]
            last_x = state_history[-3, 0]
            last_y = state_history[-3, 1]
        
        #if step_index < 15:
        #    k = -0.005 * (last_gamma - 0)
        #else:
        #    k = -0.005 * (last_gamma - np.pi/2)
        
        last_gamma_cos = np.cos(last_gamma)
        last_gamma_sin = np.sin(last_gamma)
        
        
        last_x_pixels = int(np.round((last_x) / RESOLUTION))
        last_y_pixels = int(np.round((last_y) / RESOLUTION))
        condition = EDGE_MARGIN <= last_x_pixels
        condition = condition and last_x_pixels <= vx.shape[1] - 1 - EDGE_MARGIN
        condition = condition and EDGE_MARGIN <= last_y_pixels
        condition = condition and last_y_pixels <= vx.shape[0] - 1 - EDGE_MARGIN
        if condition:
            residual_tmp = field_dist[last_y_pixels, last_x_pixels]
        else:
            residual_tmp = 2738.5864 # np.max(field_dist)
        residual +=  residual_tmp/ 2000.0
        target_direction_x, target_direction_y, condition = get_target_direction(last_x, last_y, vx, vy)
        if condition:
            
            #dot_prod = target_direction_x * last_gamma_cos + target_direction_y * last_gamma_sin
            #angle_delta = np.arccos(dot_prod)
            #k = -0.0005 * angle_delta
            
            pseudo_dot_prod = target_direction_x * last_gamma_sin - target_direction_y * last_gamma_cos
            if pseudo_dot_prod > 1.0:
                pseudo_dot_prod = 1.0
            if pseudo_dot_prod < -1.0:
                pseudo_dot_prod = -1.0
            angle_delta = np.arcsin(pseudo_dot_prod)
            #k = -loopback_stength * angle_delta
            U = U_old + k_p_tmp * (angle_delta - angle_delta_old) + k_i_tmp  * angle_delta + k_d_tmp  * (angle_delta - 2 * angle_delta_old + angle_delta_old_2)
            
            v = 150.0
            dL = v * dt
            #dgamma = k * dL
            dgamma = -U * dL
            
            #print('k =', k, '  U =', U)
            
            #if step_index > 3:
            #    import sys
            #    sys.exit()
            
            for time_index in range(times.size):
                state[0] += dL * np.cos(state[2])
                state[1] += dL * np.sin(state[2])
                state[2] += dgamma
                state_history_detailed = np.concatenate((state_history_detailed, state.reshape(1, 3)), axis=0)
            
            angle_delta_old_2 = angle_delta_old
            angle_delta_old = angle_delta
            
            U_old_2 = U_old
            U_old = U
        
        else:
            for time_index in range(times.size):
                state[0] += 0.0
                state[1] += 0.0
                state[2] += 0.0
                state_history_detailed = np.concatenate((state_history_detailed, state.reshape(1, 3)), axis=0)
        
                
        state_history = np.concatenate((state_history, state.reshape(1, 3)), axis=0)
        
        
    if residual < residual_min:
        residual_min = residual
        k_p_min = k_p_tmp
        k_i_min = k_i_tmp
        k_d_min = k_d_tmp   
    
    step = 200
    plt.quiver(X_pixels[::step, ::step], Y_pixels[::step, ::step], vx[::step, ::step], -vy[::step, ::step])
    
    
    
    plt.plot(state_history_detailed[:, 0], state_history_detailed[:, 1], 'k-')
    plt.plot(state_history[:, 0], state_history[:, 1], 'r.')
    
    #plt.xlim([0, 4000.0])
    #plt.ylim([0, 4000.0])
    
    margin_tmp = 200
    plt.plot([-margin_tmp, 4000 + margin_tmp, 4000 + margin_tmp, -margin_tmp, -margin_tmp], [-margin_tmp, -margin_tmp, 4000 + margin_tmp, 4000 + margin_tmp, -margin_tmp], 'k-')
    
    str_tmp = r'$k_{p} = ' + '%.2E' % Decimal(k_p_tmp) + '\ \ ' + r'k_{i} = ' + '%.2E' % Decimal(k_i_tmp) + '\ \ ' + r'k_{d} = ' + '%.2E' % Decimal(k_d_tmp) + '$'
    plt.text(-200, 100, str_tmp)
    
    str_tmp = r'$r = ' + '%.2E' % Decimal(residual) +'$'
    plt.text(-200, 300, str_tmp)
    
    plt.gca().invert_yaxis()
    plt.axis('equal')
    
    canvas = FigureCanvas(fig)
    canvas.draw()
    mat = np.array(canvas.renderer._renderer)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    
    if frame_index == 0:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
        video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (mat.shape[1], mat.shape[0])) 
    video_writer.write(mat)
video_writer.release()

print('residual_min =', residual_min)
print('k_p_min =', k_p_min)
print('k_i_min =', k_i_min)
print('k_d_min =', k_d_min)