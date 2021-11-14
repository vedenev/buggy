import numpy as np
import matplotlib.pyplot as plt
import pickle

PATH_TO_SAVE = 'tt053_saves'

y1_strip = 460
y2_strip = 570

PIXELS_TO_ANGLE = np.load('PIXELS_TO_ANGLE_2.npy')
PIXELS_TO_ANGLES_TAN_X = np.load('PIXELS_TO_ANGLES_TAN_X_2.npy')
PIXELS_TO_ANGLES_TAN_Y = np.load('PIXELS_TO_ANGLES_TAN_Y_2.npy')

PIXELS_TO_ANGLE = PIXELS_TO_ANGLE[y1_strip: y2_strip, :]
PIXELS_TO_ANGLES_TAN_X = PIXELS_TO_ANGLES_TAN_X[y1_strip: y2_strip, :]
PIXELS_TO_ANGLES_TAN_Y = PIXELS_TO_ANGLES_TAN_Y[y1_strip: y2_strip, :]


with open(PATH_TO_SAVE + '/' + 'x_detected_all_all.pickle', 'rb') as handle:
    x_detected_all_all = pickle.load(handle)
with open(PATH_TO_SAVE + '/' + 'y_detected_all_all.pickle', 'rb') as handle:
    y_detected_all_all = pickle.load(handle)
with open(PATH_TO_SAVE + '/' + 'for_localization.pickle', 'rb') as handle:
    for_localization = pickle.load(handle)

index = 35

x_detected_all = x_detected_all_all[index]
y_detected_all = y_detected_all_all[index]
for_localization_tmp = for_localization[index]
x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all = for_localization_tmp

x_detected_pairs_all_flatten = x_detected_pairs_all.flatten()
y_detected_pairs_all_flatten = y_detected_pairs_all.flatten()
tan_x_flatten = PIXELS_TO_ANGLES_TAN_X[y_detected_pairs_all_flatten, x_detected_pairs_all_flatten]
tan_y_flatten = PIXELS_TO_ANGLES_TAN_Y[y_detected_pairs_all_flatten, x_detected_pairs_all_flatten]

plt.subplot(2, 1, 1)
plt.plot(x_detected_pairs_all_flatten, y_detected_pairs_all_flatten, 'k.')
#plt.axis('equal')

plt.subplot(2, 1, 2)
plt.plot(tan_x_flatten, tan_y_flatten, 'k.')
#plt.axis('equal')

plt.show()


