from code_detector_2020_08_24 import detect_codes, y1_strip, y2_strip, WIDTH, HEIGHT
from localize_2020_09_03 import localize
import cv2
import numpy as np
import time
import glob
import os
import wiringpi
import threading

UPDATE_WHEELS_VALUES_PERIOD = 0.3
PRE_PHOTO_PAUSE = 0.5
APHTER_PHOTO_PAUSE = 0.3
CLOSE_TO_WALL_SKEEP_THRESHOLD = 2

MIN_WALL_DISTANCE = 1200.0

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
MINIMAL_MOVABLE_VALUE_STRIGHT_RIGHT = 375
MINIMAL_MOVABLE_VALUE_STRIGHT_LEFT = 375


ANGLE_SMOOTH_CONTROL_THRESHOLD = np.pi / 6
SMOOTH_CONTROL_BASE_VALUE = 370
SMOOTH_CONTROL_KP = 150.0

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


counter_right = 0
def encoder_right_callback():
    global counter_right
    #print("er" + str(counter_right))
    counter_right += 1


counter_left = 0
def encoder_left_callback():
    global counter_left
    #print("el" + str(counter_left))
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

stop_flag = False
is_localized = False
current_solution  = np.zeros((3, 1), np.float32)
current_solution[0, 0] = 0.0
current_solution[1, 0] = 2000.0
current_solution[2, 0] = -np.pi/2

set_wheeles_lock = threading.Lock()

right_value_global = 0
left_value_global = 0

close_to_wall = False

def correct_value(value):
    if value > 1023:
        value = 1023
    if value < -1023:
        value = -1023
    return value

def int_round(inp):
    return int(np.round(inp))

def set_wheeles(right_value, left_value):
    global set_wheeles_lock
    right_value = int_round(right_value)
    left_value = int_round(left_value)
    right_value = correct_value(right_value)
    left_value = correct_value(left_value)
    with set_wheeles_lock:
        if right_value == 0:
            wiringpi.pwmWrite(pin_PWM_right, 0)
            wiringpi.digitalWrite(pin_forward_right, 0) 
            wiringpi.digitalWrite(pin_backward_right, 0)
        else:
            right_value_abs = abs(right_value)
            wiringpi.pwmWrite(pin_PWM_right, right_value_abs)
            if right_value > 0:
                wiringpi.digitalWrite(pin_forward_right, 1) 
                wiringpi.digitalWrite(pin_backward_right, 0)
            else:
                wiringpi.digitalWrite(pin_forward_right, 0) 
                wiringpi.digitalWrite(pin_backward_right, 1)

        if left_value == 0:
            wiringpi.pwmWrite(pin_PWM_left, 0)
            wiringpi.digitalWrite(pin_forward_left, 0) 
            wiringpi.digitalWrite(pin_backward_left, 0)
        else:
            left_value_abs = abs(left_value)
            wiringpi.pwmWrite(pin_PWM_left, left_value_abs)
            if left_value > 0:
                wiringpi.digitalWrite(pin_forward_left, 1) 
                wiringpi.digitalWrite(pin_backward_left, 0)
            else:
                wiringpi.digitalWrite(pin_forward_left, 0) 
                wiringpi.digitalWrite(pin_backward_left, 1)

def update_wheels_values_periodically():
    print('update_wheels_values_periodically()')
    global UPDATE_WHEELS_VALUES_PERIOD
    global is_localized
    global current_solution
    global stop_flag
    global right_value_global
    global left_value_global
    global close_to_wall
    
    x = current_solution[0, 0]
    y = current_solution[1, 0]
    angle = current_solution[2, 0]
    if is_localized:
        if y < MIN_WALL_DISTANCE:
            close_to_wall = True
    
    if close_to_wall:
        right_value = 0
        left_value = 0
    else:  # y >= MIN_WALL_DISTANCE
        if is_localized:
            if np.sin(angle) > np.sin(-np.pi/2 + ANGLE_SMOOTH_CONTROL_THRESHOLD):
                # constant rotation
                if np.cos(angle) > 0:
                    right_value = 0
                    left_value = MINIMAL_MOVABLE_VALUE_LEFT
                else:
                    right_value = MINIMAL_MOVABLE_VALUE_RIGHT
                    left_value = 0
            else:
                # smooth control
                #SMOOTH_CONTROL_BASE_VALUE = 370
                #SMOOTH_CONTROL_KP = 35.0
                error = np.cos(angle) - np.cos(-np.pi / 2)
                right_value = SMOOTH_CONTROL_BASE_VALUE - int_round(SMOOTH_CONTROL_KP * error)
                left_value = SMOOTH_CONTROL_BASE_VALUE + int_round(SMOOTH_CONTROL_KP * error)
        else:
            # rotate:
            right_value = 0
            left_value = MINIMAL_MOVABLE_VALUE_LEFT
        
        
    if stop_flag:
        set_wheeles(0, 0)
    else:
        set_wheeles(right_value, left_value)
        right_value_global = right_value
        left_value_global = left_value
    
    threading.Timer(UPDATE_WHEELS_VALUES_PERIOD, update_wheels_values_periodically).start()

camera_capturer = cv2.VideoCapture(0)

if not (camera_capturer.isOpened()):
    print('Could not open video device')
    import sys
    sys.exit()

camera_capturer.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera_capturer.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

close_to_wall_counter = 0
update_wheels_values_periodically()

frame_index = 0
#n_frames = 100
#time_1 = time.time()
#for frame_index in range(n_frames):
while True:
    stop_flag = True
    set_wheeles(0, 0)
    time.sleep(PRE_PHOTO_PAUSE) 
    is_frame_read, frame = camera_capturer.read()
    stop_flag = False
    set_wheeles(right_value_global, left_value_global)
    
    if not is_frame_read:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    ## for debug:
    #frame = cv2.imread('./photos_for_localization_test/028.png')
    time_1 = time.time()
    x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all = detect_codes(np.copy(frame))
    time_2 = time.time()
    time_detect = time_2 - time_1
    if time_detect < APHTER_PHOTO_PAUSE:
        time.sleep(APHTER_PHOTO_PAUSE - time_detect)
    is_localized, current_solution = localize(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all)
    #if not is_localized:
    #    current_solution  = np.zeros((3, 1), np.float32)
    #    current_solution[0, 0] = 0.0
    #    current_solution[1, 0] = 2000.0
    #    current_solution[2, 0] = -np.pi/2
    
    if not is_localized:
        if close_to_wall_counter >= CLOSE_TO_WALL_SKEEP_THRESHOLD:
            close_to_wall = False
            close_to_wall_counter = 0
        else:
            close_to_wall_counter += 1
        
    
    print(' ')
    print('is_localized =', is_localized)
    print('current_solution =', current_solution)
    
    if frame_index % 5 == 0:
        cv2.imwrite('./tt0037_tmp' + '/' + str(frame_index).zfill(4) + '.png', frame)
    
    frame_index += 1
    
#time_2 = time.time()
#print((time_2 - time_1) / n_frames) # 0.3109789037704468
    
camera_capturer.release()