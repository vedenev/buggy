import numpy as np
import matplotlib.pyplot as plt




speed_right_all = np.load('tt014_speed_right_all.npy')
speed_left_all = np.load('tt014_speed_left_all.npy')
speed_right_target_all = np.load('tt014_speed_right_target_all.npy')
speed_left_target_all = np.load('tt014_speed_left_target_all.npy')
right_value_all = np.load('tt014_right_value_all.npy')
left_value_all = np.load('tt014_left_value_all.npy')

plt.plot(speed_right_all, 'r^-', label='speed_right_all')
plt.plot(speed_left_all, 'rv-', label='speed_left_all')
plt.plot(speed_right_target_all, 'g^-', label='speed_right_target_all')
plt.plot(speed_left_target_all, 'gv-', label='speed_left_target_all')
plt.plot(right_value_all, 'b^-', label='right_value_all')
plt.plot(left_value_all, 'bv-', label='left_value_all')
plt.legend()
plt.show()