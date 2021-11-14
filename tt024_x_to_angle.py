#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:45:52 2020

@author: vedenev
"""

import numpy as np
import matplotlib.pyplot as plt 

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

#r2 = xs**2
#xss = xs * (1 + k1 * r2 + k2 * r4 + k3 * r6) + p2 * (r2 + 2 * xs**2)
#xss = xs * (1 + k1 * xs**2 + k2 * xs**4 + k3 * xs**6) + p2 * (xs**2 + 2 * xs**2)
#xss = xs * (1 + 3*p2*xs +  k1 * xs**2  + k2 * xs**4 + k3 * xs**6)
a = 3*p2
b = k1
c = k2
d = k3


# install maxima in Ubuntu: sudo apt-get install maxima
#load ("revert")$
#t: taylor (x*(1 + a * x + b*x**2 + c*x**4 + d*x**6), x, 0, 9);
#revert2 (t, x, 11);
#result:
#                   2         2           4            2          2
#      ((12 c - 78 b  + 1092 a  b - 1365 a ) d + (546 a  - 78 b) c
#         3         2  2          4            6           5          2  4
# + (364 b  - 8190 a  b  + 21840 a  b - 12376 a ) c - 273 b  + 10920 a  b
#          4  3           6  2          8            10   11
# - 61880 a  b  + 111384 a  b  - 75582 a  b + 16796 a  ) x
#          3                      2           2         3           5
# + ((286 a  - 132 a b) d - 66 a c  + (858 a b  - 4004 a  b + 3003 a ) c
#           4          3  3          5  2          7           9   10
# - 1001 a b  + 10010 a  b  - 24024 a  b  + 19448 a  b - 4862 a ) x
#                2         2           2         2          4          4
# + ((10 b - 55 a ) d + 5 c  + ((- 55 b ) + 660 a  b - 715 a ) c + 55 b
#         2  3         4  2         6           8   9
# - 1430 a  b  + 5005 a  b  - 5005 a  b + 1430 a ) x
#                  3                      3        3  2         5          7   8
# + (9 a d + (165 a  - 90 a b) c + 165 a b  - 990 a  b  + 1287 a  b - 429 a ) x
#                       2          3        2  2        4          6   7
# + ((- d) + (8 b - 36 a ) c - 12 b  + 180 a  b  - 330 a  b + 132 a ) x
#                  2       3         5   6               2       2         4   5
# + (7 a c - 28 a b  + 84 a  b - 42 a ) x  + ((- c) + 3 b  - 21 a  b + 14 a ) x
#               3   4       2       3      2
# + (5 a b - 5 a ) x  + (2 a  - b) x  - a x  + x



ai = -a
bi = (2*a*2 - b)
ci = (5 * a * b - 5 * a**3)
di = (-c + 3 * b**2 - 21 * a**2 * b + 14 * a**4)
ei = (7 * a * c - 28 * a * b**2 + 84 * a**3 * b - 42 * a**5)
fi = (-d + (8 * b - 36 * a**2)*c - 12 * b**3 + 180 * a**2 * b**2 - 330 * a**4 * b + 132 * a**6)
gi = (9 * a * d + (165 * a**3 - 90 * a* b) * c + 165 * a * b**3 - 990 * a**3 * b**2 + 1287 * a**5 * b - 429 * a**7)

alpha = np.linspace(-np.pi / 3, np.pi / 3, 101)
   
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


#ui = np.linspace(0, 1280, 101)
ui = u
xssi = (ui - cx) / fx
xsi = xssi * (1 + ai * xssi + bi * xssi**2 + ci * xssi**3 + di * xssi**4 + ei * xssi**5 + fi * xssi**6 + gi * xssi**7)
alphai = np.arctan(xsi)
alphai_linear = 1.02 * xssi
plt.subplot(1, 2, 1)
plt.plot(u, (180 / np.pi) * alpha, 'r-')
plt.plot(ui, (180 / np.pi) * alphai, 'k-')
#plt.plot(ui, alphai_linear, 'g-')

plt.subplot(1, 2, 2)
plt.plot(ui, (180 / np.pi) * (alpha - alphai), 'k-')
#plt.plot(ui, alpha - alphai_linear, 'g-')

