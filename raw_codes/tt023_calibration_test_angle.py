#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:12:03 2020

@author: vedenev
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt 

def imshow_bgr(img_brg):
    img_rgb = cv2.cvtColor(img_brg, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)

mtx = np.array([[617.8249050804442, 0.0, 673.0536941293645], [0.0, 619.3492046143635, 497.9661474464693], [0.0, 0.0, 1.0]])
dist = np.array([[-0.3123562037471547, 0.1018281655721802, 0.00031297833728767365, 0.0007424882126541622, -0.015160446251882953]])

k1 = dist[0, 0]
k2 = dist[0, 1]
p1 = dist[0, 2]
p2 = dist[0, 3]
k3 = dist[0, 4]

fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]

img = cv2.imread('./photos_for_calibration/052.png')

#alpha = np.linspace(-np.pi / 3, np.pi / 3, 13)
alpha = np.linspace(-np.pi / 3, np.pi / 3, 25)
   
#xs = x / z
# tan(alpha) = x / z = xs

xs = np.tan(alpha)
ys = 0.0

r2 = xs ** 2 + ys ** 2
r4 = r2 ** 2
r6 = r4 * r2
xss = xs * (1 + k1 * r2 + k2 * r4 + k3 * r6) + p2 * (r2 + 2 * xs**2)
yss = p1 * (r2 + 2 * ys**2)

u = fx * xss + cx
v = fy * yss + cy

#imshow_bgr(img)
#plt.plot(u, v, 'r.')

plt.plot(u, alpha, 'r.-')

alpha_fit = 0.00164 * (u - 673)
plt.plot(u, alpha_fit, 'k--')