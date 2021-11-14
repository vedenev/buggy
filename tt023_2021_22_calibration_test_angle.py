#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:12:03 2020

@author: vedenev
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt 

h = 960
w = 1280

def imshow_bgr(img_brg):
    img_rgb = cv2.cvtColor(img_brg, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)

mtx=np.array([[618.7162038517924, 0.0, 671.9856848350533], [0.0, 620.20895334707, 497.72866882850883], [0.0, 0.0, 1.0]])
dist=np.array([[-0.3132964900492584, 0.10279928422337736, 0.0003079750471391313, 0.0007692445996596191, -0.015460211904764277]])

k1 = dist[0, 0]
k2 = dist[0, 1]
p1 = dist[0, 2]
p2 = dist[0, 3]
k3 = dist[0, 4]

#k4 = dist[0, 5]
#k5 = dist[0, 6]
#k6 = dist[0, 7]
#s1 = dist[0, 8]
#s2 = dist[0, 9]
#s3 = dist[0, 10]
#s4 = dist[0, 11]

fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]

img = cv2.imread('./photos_for_calibration/052.png')


mx = 1.2
my = 1.1
alpha_x_0 = np.linspace(-mx * np.pi / 3, mx * np.pi / 3, 35)
alpha_y_0 = np.linspace(-my * np.pi / 3, my * np.pi / 3, 75)
alpha_x, alpha_y = np.meshgrid(alpha_x_0, alpha_y_0)
   
#xs = x / z
# tan(alpha) = x / z = xs

xs = np.tan(alpha_x)
ys = np.tan(alpha_y)

r2 = xs ** 2 + ys ** 2
r4 = r2 ** 2
r6 = r4 * r2

#m = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
#xss = xs * m + 2 * p1 * xs * ys + p2 * (r2 + 2 * xs**2) + s1 * r2 + s2 * r4
#yss = ys * m + p1 * (r2 + 2 * ys**2) + 2 * p2 * xs * ys  + s3 * r2 + s4 * r4

m = (1 + k1 * r2 + k2 * r4 + k3 * r6)
xss = xs * m + 2 * p1 * xs * ys + p2 * (r2 + 2 * xs**2)
yss = ys * m + p1 * (r2 + 2 * ys**2) + 2 * p2 * xs * ys

u = fx * xss + cx
v = fy * yss + cy

plt.plot(u.flatten(), v.flatten(), 'k.')
plt.plot([0, w, w, 0, 0], [0, 0, h, h, 0], 'g-')
plt.show()