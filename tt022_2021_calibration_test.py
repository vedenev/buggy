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

mtx=np.array([[613.6206796762666, 0.0, 683.4888941175157], [0.0, 615.0136787931489, 499.2466309019339], [0.0, 0.0, 1.0]])
dist=np.array([[-1.047488351559334, 0.1895766552719301, 0.0005408151134692955, 0.00010288485616716016, 0.053297712776547455, -0.707423458071405, -0.2169199947848843, 0.17741744871291024, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix= mtx, distCoeffs=dist, imageSize=(w,h), alpha=0.2, newImgSize=(w,h),centerPrincipalPoint=1)
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)


img = cv2.imread('./photos_for_calibration/052.png')

img2 = cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)

imshow_bgr(img2)