# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 07:27:45 2019

@author: vedenev
"""

import cv2
import numpy as np

CODE = 123
SQURE_SIZE = 200
MARGIN_SIZE_RELATIVE = 0.1

margin_size = int(SQURE_SIZE * MARGIN_SIZE_RELATIVE)

code_binary_str = '{0:08b}'.format(CODE)
code_binary = [x == '1' for x in code_binary_str]

code_2d = np.zeros((2, 7), np.bool)
code_2d[0,0] = code_binary[0]
code_2d[1,1] = code_2d[0,0]
code_2d[0,1] = not(code_2d[0,0])
code_2d[1,0] = not(code_2d[0,0])
code_2d[0,-2] = code_binary[1]
code_2d[1,-1] = code_2d[0,-2]
code_2d[0,-1] = not(code_2d[0,-2])
code_2d[1,-2] = not(code_2d[0,-2])

count = 2
for y in range(code_2d.shape[0]):
    for x in range(2, code_2d.shape[1]-2):
        code_2d[y, x] = code_binary[count]
        count += 1

image = np.full((2 * SQURE_SIZE + 2 * margin_size, code_2d.shape[1] * SQURE_SIZE + 2 * margin_size, 3), 255 ,dtype=np.uint8)

for y in range(code_2d.shape[0]):
    y1 = margin_size + y * SQURE_SIZE
    y2 = margin_size + (y + 1) * SQURE_SIZE
    for x in range(code_2d.shape[1]):
        x1 = margin_size + x * SQURE_SIZE
        x2 = margin_size + (x + 1) * SQURE_SIZE
        
        if code_2d[y, x]:
            image[y1: y2, x1: x2, :] = 0
            

cv2.imwrite('code_' + str(CODE) + '.png', image)

