#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 17:22:38 2020

@author: vedenev
"""

import glob
import os
from code_detector_2020_08_24 import detect_codes, y1_strip, y2_strip, WIDTH, HEIGHT 
import cv2
import numpy as np


SELECTED = [17, 20, 24, 28, 31, 36, 40, 43, 47, 55, 58, 62, 66]

DIR = './photos_for_localization_test'
OUTPUT_DIR = './photos_for_localization_test_visualize'

X_TO_ANGLE_X_LIMITS = [90, 1270]
X_TO_ANGLE_TAILOR_COEFFS = \
    [-0.002227464637962487, 0.3212660622990047, -0.003478867250403373, 0.1909035743393775, -0.0044976791528199165, 0.12648175837018866, -0.005129167400927317]
X_TO_ANGLE_FX = 617.8249050804442
X_TO_ANGLE_CX = 673.0536941293645

def select_codes_in_limits(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all):
    
    
    if x_detected_all.size > 0:
        condition = (X_TO_ANGLE_X_LIMITS[0] <= x_detected_pairs_all) & (x_detected_pairs_all <= X_TO_ANGLE_X_LIMITS[1])
        condition_codewise = np.all(condition, axis=1)
        
        x_detected_all = x_detected_all[condition_codewise]
        y_detected_all = y_detected_all[condition_codewise]
        code_detected_all = code_detected_all[condition_codewise]
        code_size_all = code_size_all[condition_codewise]
        x_detected_pairs_all = x_detected_pairs_all[condition_codewise, :]
        y_detected_pairs_all = y_detected_pairs_all[condition_codewise, :]
    
    return x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all

def x_to_angle(x):
    xss = (x - X_TO_ANGLE_CX) / X_TO_ANGLE_FX
    xs = np.copy(xss)
    power = np.copy(xss)
    for monom_index in range(len(X_TO_ANGLE_TAILOR_COEFFS)):
        power *= xss
        xs += X_TO_ANGLE_TAILOR_COEFFS[monom_index] * power
    angles = np.arctan(xs)
    return angles
        

files = sorted(glob.glob(DIR + '/*.png'))
lc = 0
for index in range(len(files)):
    file_ = files[index]
    file_base = os.path.basename(file_)
    file_base_no_ext = os.path.splitext(file_base)[0]
    if len(file_base_no_ext) == 3:
        file_no = int(file_base_no_ext)
        if file_no in SELECTED:
            frame = cv2.imread(file_)
            x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all = detect_codes(np.copy(frame))
            
            x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all = \
                select_codes_in_limits(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all)
            
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            if x_detected_all.size >= 1:
                angles_detected_pairs_all = x_to_angle(x_detected_pairs_all)
                angles_detected_pairs_all =  -angles_detected_pairs_all[:,::-1] # because upsidedown
                
                y_detected_all = y1_strip + y_detected_all 
                x_detected_all = WIDTH - x_detected_all
                y_detected_all = HEIGHT - y_detected_all
                
                
                y_detected_pairs_all = y1_strip + y_detected_pairs_all
                x_detected_pairs_all = WIDTH - x_detected_pairs_all
                y_detected_pairs_all = HEIGHT - y_detected_pairs_all
                
                
                
                for code_index in range(x_detected_all.size):
                    cv2.putText(frame, str(code_detected_all[code_index]), (int(x_detected_all[code_index]), int(y_detected_all[code_index])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.circle(frame, (int(x_detected_pairs_all[code_index, 0]), int(y_detected_pairs_all[code_index, 0])), 3, (255, 0, 255), -1) 
                    cv2.circle(frame, (int(x_detected_pairs_all[code_index, 1]), int(y_detected_pairs_all[code_index, 1])), 3, (255, 0, 255), -1) 
            
                # sort by 1st point
                sort_indexes = np.argsort(angles_detected_pairs_all[:, 0])
                angles_detected_pairs_all = angles_detected_pairs_all[sort_indexes, :]
                code_detected_all = code_detected_all[sort_indexes]
                
                angular_sizes_detected_all = angles_detected_pairs_all[:, 1] - angles_detected_pairs_all[:, 0]
                
                
            
            cv2.imwrite(OUTPUT_DIR + '/' + file_base, frame)
        