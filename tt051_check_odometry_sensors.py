from code_detector_2020_08_24 import detect_codes, y1_strip, y2_strip, WIDTH, HEIGHT
from localize_with_circles_2020_12_29 import localize_with_circles, codes_centres,  POSITIONS_CODES_REAL
from prepare_vector_field import vx, vy, RESOLUTION, TARGET_X, TARGET_Y, TARGET_TRAJECTORY, X, Y
import cv2
import numpy as np
import time
import glob
import os
import wiringpi
import threading
from picamera.array import PiRGBArray
from picamera import PiCamera
import pickle

# turn formulas:
N_holes = 20
r_wheel = 31 # mm
a_wheels = 118 # mm, half-distance between wheels
pi_r_N = np.pi * r_wheel / N_holes
pi_r_N_a = pi_r_N  / a_wheels
# rotation anlge: (2 * np.pi * r_wheel / (N_holes * 2 * a_wheels)) * (i - j)
# rotation radius: R = a_wheels * (i + j) / (i - j)

## because in last point probably don't see 2 codes
#TARGET_X = 2531
#TARGET_Y = 4000

MANUAL_BALANCE_COEEFICIENT = -0.05;

JUST_DRAW = False;

PATH_TO_SAVE = 'tt050_saves'

MIN_TARGET_DISTANCE = 500.0 ** 2

UPDATE_WHEELS_VALUES_PERIOD = 0.03
PRE_PHOTO_PAUSE = 0.3
APHTER_PHOTO_PAUSE = 0.14
CLOSE_TO_TARGET_SKEEP_THRESHOLD = 10


#MINIMAL_MOVABLE_VALUE_RIGHT = 370
#MINIMAL_MOVABLE_VALUE_LEFT = 375
#MINIMAL_MOVABLE_VALUE_STRIGHT_RIGHT = 350
#MINIMAL_MOVABLE_VALUE_STRIGHT_LEFT = 355

#MINIMAL_MOVABLE_VALUE_RIGHT = 375
#MINIMAL_MOVABLE_VALUE_LEFT = 375
#MINIMAL_MOVABLE_VALUE_STRIGHT_RIGHT = 355
#MINIMAL_MOVABLE_VALUE_STRIGHT_LEFT = 355

# increase for impulses mode:
MINIMAL_MOVABLE_VALUE_RIGHT = 395
MINIMAL_MOVABLE_VALUE_LEFT = 395


ANGLE_SMOOTH_CONTROL_THRESHOLD = np.pi / 4
SMOOTH_CONTROL_BASE_VALUE = 400
SMOOTH_CONTROL_KP = 70.0

pin_PWM_right = 1
pin_forward_right = 11
pin_backward_right = 10

pin_PWM_left = 23
pin_forward_left = 13
pin_backward_left = 12

DIGITAL_OUT_MODE = 1
PWM_MODE = 2

pin_encoder_right = 0
pin_encoder_left = 2

ANGLE_SMOOTH_CONTROL_THRESHOLD_COS = np.cos(ANGLE_SMOOTH_CONTROL_THRESHOLD)

if JUST_DRAW:
    import matplotlib.pyplot as plt 

    step = 30
    plt.quiver(X[::step, ::step] / RESOLUTION, Y[::step, ::step] / RESOLUTION, vx[::step, ::step], vy[::step, ::step])
    plt.plot(TARGET_TRAJECTORY[:, 0], TARGET_TRAJECTORY[:, 1], 'g-')
    plt.plot(TARGET_X, TARGET_Y, 'dr')
    
    n_codes = codes_centres.shape[0]
    for point_index in range(n_codes):
        position_code_real = POSITIONS_CODES_REAL[point_index, :, :]
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
    
    print('POSITIONS_CODES_REAL =', POSITIONS_CODES_REAL)
    
    plt.show()
    
    import sys
    sys.exit()


counter_right = 0
def encoder_right_callback():
    global counter_right
    print("er" + str(counter_right))
    counter_right += 1

counter_left = 0
def encoder_left_callback():
    global counter_left
    print("el" + str(counter_left))
    counter_left += 1

wiringpi.wiringPiSetup()

# set PWM mode:
wiringpi.pinMode(pin_PWM_right, PWM_MODE)
wiringpi.pinMode(pin_PWM_left, PWM_MODE)

# set digital output mode:
wiringpi.pinMode(pin_forward_right, DIGITAL_OUT_MODE)
wiringpi.pinMode(pin_backward_right, DIGITAL_OUT_MODE)
wiringpi.pinMode(pin_forward_left, DIGITAL_OUT_MODE)
wiringpi.pinMode(pin_backward_left, DIGITAL_OUT_MODE)

# set digital input mode:
#wiringpi.wiringPiSetupGpio()
wiringpi.pinMode(pin_encoder_right, wiringpi.GPIO.INPUT)
wiringpi.pullUpDnControl(pin_encoder_right, wiringpi.GPIO.PUD_UP)
wiringpi.pinMode(pin_encoder_left, wiringpi.GPIO.INPUT)
wiringpi.pullUpDnControl(pin_encoder_left, wiringpi.GPIO.PUD_UP)

wiringpi.wiringPiISR(pin_encoder_right, wiringpi.GPIO.INT_EDGE_BOTH, encoder_right_callback)
wiringpi.wiringPiISR(pin_encoder_left, wiringpi.GPIO.INT_EDGE_BOTH, encoder_left_callback)

stop_flag = True
is_localized = False
current_solution  = np.zeros((3, 1), np.float32)
current_solution[0, 0] = np.NaN
current_solution[1, 0] = np.NaN
current_solution[2, 0] = np.NaN


right_value_global = 0
left_value_global = 0

right_value = 0
left_value = 0

time.sleep(10)
