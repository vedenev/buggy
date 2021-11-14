# -*- coding: utf-8 -*-
"""
Created on Fri May  3 17:57:14 2019

@author: vedenev
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt 

def imshow_bgr(img_brg):
    img_rgb = cv2.cvtColor(img_brg, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)

h = 960
w = 1280

mtx=np.array([[617.8249050804442, 0.0, 673.0536941293645], [0.0, 619.3492046143635, 497.9661474464693], [0.0, 0.0, 1.0]])
dist=np.array([[-0.3123562037471547, 0.1018281655721802, 0.00031297833728767365, 0.0007424882126541622, -0.015160446251882953]])

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix= mtx, distCoeffs=dist, imageSize=(w,h), alpha=0.2, newImgSize=(w,h),centerPrincipalPoint=1)
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)


img = cv2.imread('./photos_for_calibration/052.png')

img2 = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

imshow_bgr(img2)