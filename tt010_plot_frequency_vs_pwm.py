import numpy as np
import matplotlib.pyplot as plt

time_index_all = np.abs(np.load('time_index_all.npy'))
speed_right_all = np.load('speed_right_all.npy', )
speed_left_all = np.load('speed_left_all.npy')

plt.plot(time_index_all, speed_right_all, 'r.-', label='right')
plt.plot(time_index_all, speed_left_all, 'b.-', label='left')
plt.xlabel('PWM')
plt.ylabel('speed')
plt.legend()
plt.show()

# 344, 90
# 506, 161

k = (161 - 90) / (506 - 344)
ki = 1/k
print("ki =", ki) # ki = 2.2816901408450705 PWM/speed