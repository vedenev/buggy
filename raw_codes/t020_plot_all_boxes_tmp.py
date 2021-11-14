# -*- coding: utf-8 -*-
"""
Created on Thu May  7 15:05:19 2020

@author: vedenev
"""

import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

HEIGHT = 960
WIDTH = 1280

Y1 = 460
Y2 = 570


plt.plot([0, WIDTH, WIDTH, 0, 0], [0, 0, HEIGHT, HEIGHT, 0], 'k-')
DIR = './datasets/dataset_2020_05_06/sets_all'
files = glob.glob(DIR + '/*.xml')
for file_index in range(len(files)):
    file_tmp = files[file_index]
    root = ET.parse(file_tmp).getroot()

    for obj in root.findall('object'):
        
        code = int(obj.find('name').text)
        
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        
        
        plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], 'r-')
        

plt.plot([0, WIDTH], [Y1, Y1], 'g-')
plt.plot([0, WIDTH], [Y2, Y2], 'g-')

plt.gca().invert_yaxis()
plt.axis('scaled')
    