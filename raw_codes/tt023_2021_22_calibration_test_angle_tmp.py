#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:12:03 2020

@author: vedenev
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt 
import time

time_1 = time.time()

h = 960
w = 1280

def imshow_bgr(img_brg):
    img_rgb = cv2.cvtColor(img_brg, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)

mtx=np.array([[616.216941335761, 0.0, 676.4908636515407], [0.0, 617.1366312504531, 530.5681049561929], [0.0, 0.0, 1.0]])
dist=np.array([[-1.1269606631511337, 0.22228484542127555, 0.0049154225144352245, -0.0002912859369946006, 0.06317823645404994, -0.7829415613064403, -0.21837220245608102, 0.206830100550455, 0.003653985959286356, -0.0006982421696678767, -0.018892678275357754, 0.0025828333327129562]])

k1 = dist[0, 0]
k2 = dist[0, 1]
p1 = dist[0, 2]
p2 = dist[0, 3]
k3 = dist[0, 4]
k4 = dist[0, 5]
k5 = dist[0, 6]
k6 = dist[0, 7]
s1 = dist[0, 8]
s2 = dist[0, 9]
s3 = dist[0, 10]
s4 = dist[0, 11]

fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]

#img = cv2.imread('./photos_for_calibration/052.png')


mx = 1.2
my = 1.1
f = 3
alpha_x_0 = np.linspace(-mx * np.pi / 3, mx * np.pi / 3, f * w)
alpha_y_0 = np.linspace(-my * np.pi / 3, my * np.pi / 3, f * h)
alpha_x, alpha_y = np.meshgrid(alpha_x_0, alpha_y_0)
   
#xs = x / z
# tan(alpha) = x / z = xs

xs = np.tan(alpha_x)
ys = np.tan(alpha_y)

r2 = xs ** 2 + ys ** 2
r4 = r2 ** 2
r6 = r4 * r2
m = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
xss = xs * m + 2 * p1 * xs * ys + p2 * (r2 + 2 * xs**2) + s1 * r2 + s2 * r4
yss = ys * m + p1 * (r2 + 2 * ys**2) + 2 * p2 * xs * ys  + s3 * r2 + s4 * r4

u = fx * xss + cx
v = fy * yss + cy

#plt.plot(u.flatten(), v.flatten(), 'k.')
#plt.plot([0, w, w, 0, 0], [0, 0, h, h, 0], 'g-')

u = np.round(u).astype(np.int64)
v = np.round(v).astype(np.int64)

u = u.flatten();
v = v.flatten();
alpha_x = alpha_x.flatten();
xs = xs.flatten()
ys = ys.flatten()


#indexes_x, indexes_y = np.nonzero((0 <= u) & (u < w) & (0 <= v) & (v < h))

indexes = np.nonzero((0 <= u) & (u < w) & (0 <= v) & (v < h))[0]

u = u[indexes]
v = v[indexes]
alpha_x = alpha_x[indexes]
xs = xs[indexes]
ys = ys[indexes]

#plt.plot(u, v, 'k.')
uv  = u + v * w

uv_unique, uv_unique_inverse  = np.unique(uv, return_inverse=True)
indexes_tmp = np.argsort(uv_unique_inverse)
u_sorted = u[indexes_tmp]
v_sorted = v[indexes_tmp]
uv_sorted = uv[indexes_tmp]
alpha_x_sorted = alpha_x[indexes_tmp]
xs_sorted = xs[indexes_tmp]
ys_sorted = ys[indexes_tmp]
uv_unique_inverse = uv_unique_inverse[indexes_tmp]

#print(np.all(uv_unique[uv_unique_inverse] == uv_sorted)) # True

uv_unique_inverse_extended = np.zeros(uv_unique_inverse.size + 2, np.int64)
uv_unique_inverse_extended[0] = uv_unique_inverse[0] - 1
uv_unique_inverse_extended[1:-1] = uv_unique_inverse
uv_unique_inverse_extended[-1] = uv_unique_inverse[-1] + 1

diff_tmp = np.diff(uv_unique_inverse_extended)
jumps = np.nonzero(diff_tmp != 0)[0]

starts = jumps[0:-1]
ends = jumps[1:]
PIXELS_TO_ANGLE = np.full((h, w), np.nan, np.float32)
PIXELS_TO_ANGLES_TAN_X = np.full((h, w), np.nan, np.float32)
PIXELS_TO_ANGLES_TAN_Y = np.full((h, w), np.nan, np.float32)
for unique_index in range(starts.size):
    start = starts[unique_index]
    end = ends[unique_index]
    u_tmp = u_sorted[start]
    v_tmp = v_sorted[start]
    alpha_x_tmp = np.mean(alpha_x_sorted[start:end])
    PIXELS_TO_ANGLE[v_tmp, u_tmp] = alpha_x_tmp
    xs_tmp = np.mean(xs_sorted[start:end])
    PIXELS_TO_ANGLES_TAN_X[v_tmp, u_tmp] = xs_tmp
    ys_tmp = np.mean(ys_sorted[start:end])
    PIXELS_TO_ANGLES_TAN_Y[v_tmp, u_tmp] = ys_tmp

print(np.any(np.isnan(PIXELS_TO_ANGLE))) # if f = 3 then True

np.save('PIXELS_TO_ANGLE_22.npy', PIXELS_TO_ANGLE)
np.save('PIXELS_TO_ANGLES_TAN_X_22.npy', PIXELS_TO_ANGLES_TAN_X)
np.save('PIXELS_TO_ANGLES_TAN_Y_22.npy', PIXELS_TO_ANGLES_TAN_Y)

#plt.imshow(PIXELS_TO_ANGLE * 180 / np.pi)
#plt.colorbar()

time_2 = time.time()
time_delta = time_2 - time_1

print("time_delta =", time_delta)