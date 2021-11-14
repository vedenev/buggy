#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:34:48 2020

@author: vedenev
"""

import matplotlib.pyplot as plt 
import numpy as np

MM_IN_CM = 10.0

# 323 + 22 +268 + 55.5 +102 - 19 - (100.5 - 19 - 5.5) / 2 = 713.5
# 323 + 22 +268 + 55.5 +102 + 19 + 100 = 889.5
target_trajectory = [[133, 153.5], # 90 cm from M5 243.5  - 90 = 153.5
                     [254, 153.5],
                     [254, 448.25],
                     [713.5, 448.25],
                     [713.5, 67.5],
                     [889.5, 67.5]]

CODE_SIZE_REAL = 117.0 # mm, frol left marker center to right marker center
# A4 width: 210 mm
#POSITIONS_CODES_REAL[code_no, point_no, x/y] <-  shape: [n_codes, 2, 2]

# codes_centres[index] = [place_no, angle, x, y]
codes_centres = [[0, 90, 323, 329], # right edge of paper is 70 cm from exit of big room, 533 - 46 -77.5 - 70 - 21 / 2 = 329
                 [1, 90, 323, 209], # right edge of paper is 190 cm from exit of big room, 533 - 46 -77.5 - 190 - 21 / 2 = 209
                 [2, -90, 0, 189], # center of paper is in 189 cm from balcony wall
                 [3, -90, 0, 289], # center of paper is in 289 cm from balcony wall
                 [4, 0,  174, 80], # center of paper is in 80 cm from balcony wall and 149 cm from balcony right wall, 323 - 149 = 174
                 [5, 180, 204.5, 243.5], # center of paper is 96 cm from exit of big room in y direction, and in 118.5 from balcony right wall in -x direction, 323 - 118.5 = 204.5, 533 - 46 -77.5 - 70 - 96 = 243.5
                 [6, 180, 238, 533], # center of paper is in 85 cm from right balcony whall  in -x direction, on the wardrobe wall, 323-85 = 238
                 [7, 180, 334, 487]] # on left edge of exit of big room, 323 + 22/2 = 334, 533 - 46  = 487

target_trajectory = np.asarray(target_trajectory)
target_trajectory = MM_IN_CM * target_trajectory

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
    
    
    

plt.plot(target_trajectory[:, 0], target_trajectory[:, 1], 'k.-')
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
plt.axis('equal')