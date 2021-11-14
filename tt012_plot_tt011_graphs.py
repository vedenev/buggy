import numpy as np
import matplotlib.pyplot as plt

res_min = np.load('res_min.npy')
kp_min = np.load('kp_min.npy')
ki_min = np.load('ki_min.npy')
kd_min = np.load('kd_min.npy')

print('res_min =', res_min)
print('kp_min =', kp_min)
print('ki_min =', ki_min)
print('kd_min = ', kd_min)

# 1st run:
#kp = 0.5 * (1 + 0.5 * (2.0 * np.random.rand() - 1.0))
#ki = 2.0 * (1 + 0.5 * (2.0 * np.random.rand() - 1.0))
#kd = 2.0 * (1 + 0.5 * (2.0 * np.random.rand() - 1.0))
#res_min = 21.667732
#kp_min = 0.4779434475895268
#ki_min = 2.2094593534706917
#kd_min =  1.1981934673454346

# 2nd run:
#kp = 0.5 * (1 + 0.3 * (2.0 * np.random.rand() - 1.0))
#ki = 2.2 * (1 + 0.3 * (2.0 * np.random.rand() - 1.0))
#kd = 1.2 * (1 + 0.3 * (2.0 * np.random.rand() - 1.0))
#res_min = 15.378302
#kp_min = 0.6116337494817722
#ki_min = 2.4376230682295543
#kd_min =  0.9877934693840614

# 3rd run, 1000 tries:
#kp = 0.6 * (1 + 0.8 * (2.0 * np.random.rand() - 1.0))
#ki = 2.5 * (1 + 0.8 * (2.0 * np.random.rand() - 1.0))
#kd = 1.0 * (1 + 0.8 * (2.0 * np.random.rand() - 1.0))
#res_min = 10.835478
#kp_min = 0.3778117490646166
#ki_min = 2.894341756277149
#kd_min =  0.4266747863931132

# 4th run:
#kp = 0.38 * (1 + 0.1 * (2.0 * np.random.rand() - 1.0))
#ki = 2.9 * (1 + 0.1 * (2.0 * np.random.rand() - 1.0))
#kd = 0.42 * (1 + 0.1 * (2.0 * np.random.rand() - 1.0))
#res_min = 10.153725
#kp_min = 0.37732358733604954
#ki_min = 2.6305491775203635
#kd_min =  0.3844181467484491



speed_right_all = np.load('tt011_speed_right_all.npy')
speed_left_all = np.load('tt011_speed_left_all.npy')
speed_right_target_all = np.load('tt011_speed_right_target_all.npy')
speed_left_target_all = np.load('tt011_speed_left_target_all.npy')
right_value_all = np.load('tt011_right_value_all.npy')
left_value_all = np.load('tt011_left_value_all.npy')

plt.plot(speed_right_all, 'r^-', label='speed_right_all')
plt.plot(speed_left_all, 'rv-', label='speed_left_all')
plt.plot(speed_right_target_all, 'g^-', label='speed_right_target_all')
plt.plot(speed_left_target_all, 'gv-', label='speed_left_target_all')
plt.plot(right_value_all, 'b^-', label='right_value_all')
plt.plot(left_value_all, 'bv-', label='left_value_all')
plt.legend()
plt.show()