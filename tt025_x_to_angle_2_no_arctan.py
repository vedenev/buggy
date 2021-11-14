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
#t: taylor (tan(x)*(1 + a * tan(x) + b*tan(x)**2 + c*tan(x)**4 + d*tan(x)**6), x, 0, 9);
#revert2 (t, x, 11);
#result:
#                       2              2                         4
#(((170100 c - 1105650 b  + (15479100 a  - 170100) b - 19348875 a
#            2                                        2           2
# + 1105650 a  + 32130) d + ((- 1105650 b) + 7739550 a  - 85050) c
#             3                         2   2
# + (5159700 b  + (1105650 - 116093250 a ) b
#               4             2                          6             4
# + (309582000 a  - 15479100 a  + 170100) b - 175429800 a  + 19348875 a
#            2                       5               2             4
# - 1105650 a  + 35850) c - 3869775 b  + (154791000 a  - 1289925) b
#                  4              2            3
# + ((- 877149000 a ) + 38697750 a  - 368550) b
#                6              4            2           2
# + (1578868200 a  - 154791000 a  + 7739550 a  - 85050) b
#                   8               6             4            2
# + ((- 1071374850 a ) + 175429800 a  - 19348875 a  + 1105650 a  - 10329) b
#              10             8             6            4          2
# + 238083300 a   - 56388150 a  + 10319400 a  - 1289925 a  + 68466 a  - 1163)
#  11                                    3                           2
# x  )/14175 - (((1871100 a b - 4054050 a  + 155925 a) d + 935550 a c
#                   2               3                            5            3
# + ((- 12162150 a b ) + (56756700 a  - 1871100 a) b - 42567525 a  + 4054050 a
#                             4                           3   3
# - 155925 a) c + 14189175 a b  + (4054050 a - 141891750 a ) b
#               5             3              2
# + (340540200 a  - 28378350 a  + 935550 a) b
#                  7              5            3                           9
# + ((- 275675400 a ) + 42567525 a  - 4054050 a  + 155925 a) b + 68918850 a
#             7            5           3             10
# - 16216200 a  + 2837835 a  - 311850 a  + 12793 a) x  )/14175
#                  2              2            2           2                 4
# + (((90 b - 495 a  + 9) d + 45 c  + ((- 495 b ) + (5940 a  - 90) b - 6435 a
#        2               4                 2   3           4         2        2
# + 495 a  - 9) c + 495 b  + (165 - 12870 a ) b  + (45045 a  - 2970 a  + 45) b
#              6          4        2                 8         6        4
# + ((- 45045 a ) + 6435 a  - 495 a  + 9) b + 12870 a  - 3003 a  + 495 a
#       2       9                                  3                   3
# - 45 a  + 1) x )/9 + (9 a d + ((- 90 a b) + 165 a  - 9 a) c + 165 a b
#                3   2          5        3                 7       5       3
# + (45 a - 990 a ) b  + (1287 a  - 165 a  + 9 a) b - 429 a  + 99 a  - 15 a
#       8                            2              3               2   2
# + a) x  - ((7 d + ((- 56 b) + 252 a  - 7) c + 84 b  + (28 - 1260 a ) b
#          4        2               6        4       2       7
# + (2310 a  - 252 a  + 7) b - 924 a  + 210 a  - 28 a  + 1) x )/7
#                   2         3                  5       3         6
#   (21 a c - 84 a b  + (252 a  - 21 a) b - 126 a  + 28 a  - 3 a) x
# + ----------------------------------------------------------------
#                                  3
#              2         2              4       2       5
#   (5 c - 15 b  + (105 a  - 5) b - 70 a  + 15 a  - 1) x
# - -----------------------------------------------------
#                             5
#                                     2       3
#               3       4   (3 b - 6 a  + 1) x       2
# + (5 a b - 5 a  + a) x  - ------------------- - a x  + x
#                                    3
ai = -a
bi = -(3 * b - 6 * a**2 + 1) / 3
ci = (5 * a * b - 5 * a**3 + a)
di = - (5 * c - 15 * b**2 + (105 * a**2 - 5) * b - 70 * a**4 + 15 * a**2 - 1) / 5
ei = (21 * a * c - 84 * a * b**2 + (252 * a**3 - 21 * a) * b - 126 * a**5 + 28 * a**3 - 3*a) / 3
fi = -(7 * d + (- 56 * b + 252 * a**2  - 7) * c + 84 * b**3  + (28 - 1260 * a**2 ) * b**2 + (2310 * a**4  - 252 * a**2  + 7) * b - 924 * a**6  + 210 * a**4  - 28 * a**2  + 1)  / 7

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
#alpha = xssi * (1 + ai * xssi + bi * xssi**2 + ci * xssi**3 + di * xssi**4 + ei * xssi**5 + fi * xssi**6 + gi * xssi**7)
alphai = xssi * (1 + ai * xssi + bi * xssi**2 + ci * xssi**3 + di * xssi**4 + ei * xssi**5 + fi * xssi**6)
alphai_linear = 1.02 * xssi
plt.subplot(1, 2, 1)
plt.plot(u, (180 / np.pi) * alpha, 'r-')
plt.plot(ui, (180 / np.pi) * alphai, 'k-')
#plt.plot(ui, alphai_linear, 'g-')

plt.subplot(1, 2, 2)
plt.plot(ui, (180 / np.pi) * (alpha - alphai), 'k-')
#plt.plot(ui, alpha - alphai_linear, 'g-')
