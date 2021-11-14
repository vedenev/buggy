from code_detector_2020_08_24 import detect_codes, y1_strip, y2_strip, WIDTH, HEIGHT
from localize_2020_12_04 import localize, codes_centres,  POSITIONS_CODES_REAL
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

JUST_DRAW = False;

PATH_TO_SAVE = 'tt047_saves_2020_12_21'

MIN_TARGET_DISTANCE = 700.0

UPDATE_WHEELS_VALUES_PERIOD = 0.2
PRE_PHOTO_PAUSE = 0.3
APHTER_PHOTO_PAUSE = 0.3
CLOSE_TO_TARGET_SKEEP_THRESHOLD = 2


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


ANGLE_SMOOTH_CONTROL_THRESHOLD = np.pi / 4
SMOOTH_CONTROL_BASE_VALUE = 370
SMOOTH_CONTROL_KP = 50.0

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

close_to_target = False

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
    global close_to_target
    
    x = current_solution[0, 0]
    y = current_solution[1, 0]
    angle = current_solution[2, 0]
    if is_localized:
        target_distance = (TARGET_X - x)**2 +  (TARGET_Y - y)**2
        if target_distance < MIN_TARGET_DISTANCE:
            close_to_target = True
    
    if close_to_target:
        right_value = 0
        left_value = 0
    else:  # target_distance >= MIN_TARGET_DISTANCE
        is_localized_and_within_map = False
        if is_localized:
            x_px = int_round(x * RESOLUTION)
            y_px = int_round(y * RESOLUTION)
            within_map = False
            if 0 <= x_px <= vx.shape[1] - 1 and 0 <= y_px <= vx.shape[0] - 1:
                vx_tmp = vx[y_px, x_px]
                vy_tmp = vy[y_px, x_px]
                within_map = True
                print('within_map = True')
            if within_map:
                is_localized_and_within_map = True
                angle_cos = np.cos(angle)
                angle_sin = np.sin(angle)
                
                dot_product_tmp = angle_cos * vx_tmp + angle_sin * vy_tmp
                pseudo_product_tmp = angle_cos * vy_tmp - angle_sin * vx_tmp
                
                #ANGLE_SMOOTH_CONTROL_THRESHOLD_COS
                if dot_product_tmp < ANGLE_SMOOTH_CONTROL_THRESHOLD_COS:
                    # constant rotation
                    
                    if pseudo_product_tmp > 0:
                        right_value = 0
                        left_value = MINIMAL_MOVABLE_VALUE_LEFT
                    else:
                        right_value = MINIMAL_MOVABLE_VALUE_RIGHT
                        left_value = 0
                else:
                    # smooth control
                    #SMOOTH_CONTROL_BASE_VALUE = 370
                    #SMOOTH_CONTROL_KP = 35.0
                    error = np.arcsin(pseudo_product_tmp)
                    right_value = SMOOTH_CONTROL_BASE_VALUE - int_round(SMOOTH_CONTROL_KP * error)
                    left_value = SMOOTH_CONTROL_BASE_VALUE + int_round(SMOOTH_CONTROL_KP * error)
        
        print('is_localized_and_within_map =', is_localized_and_within_map)
        if not is_localized_and_within_map:
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



# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (WIDTH, HEIGHT)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(WIDTH, HEIGHT))

# allow the camera to warmup
time.sleep(1.0)

close_to_target_counter = 0
update_wheels_values_periodically()


is_localized_all = np.zeros(0, np.bool)
current_solution_all = np.zeros((0, 3), np.float32)
x_detected_all_all = []
y_detected_all_all = []
code_detected_all_all = []
for_localization = []

frame_index = 0
#n_frames = 100
#time_1 = time.time()
#for frame_index in range(n_frames):
for frame_0 in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = frame_0.array
    

    stop_flag = False
    set_wheeles(right_value_global, left_value_global)
    
    
    ## for debug:
    #frame = cv2.imread('./photos_for_localization_test/028.png')
    time_1 = time.time()
    x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all = detect_codes(np.copy(frame))
    
    is_localized, current_solution = localize(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all)
    #if not is_localized:
    #    current_solution  = np.zeros((3, 1), np.float32)
    #    current_solution[0, 0] = 0.0
    #    current_solution[1, 0] = 2000.0
    #    current_solution[2, 0] = -np.pi/2
    
    is_localized_all = np.append(is_localized_all, is_localized)
    current_solution_all = np.concatenate((current_solution_all, current_solution.reshape(1, 3)), axis=0)
    x_detected_all_all.append(x_detected_all)
    y_detected_all_all.append(y_detected_all)
    code_detected_all_all.append(code_detected_all)
    for_localization_tmp = [x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all]
    for_localization.append(for_localization_tmp)
    
    np.save(PATH_TO_SAVE + '/' + 'is_localized_all' + '.npy', is_localized_all)
    np.save(PATH_TO_SAVE + '/' + 'current_solution_all' + '.npy', current_solution_all)
    with open(PATH_TO_SAVE + '/' + 'x_detected_all_all.pickle', 'wb') as handle:
        pickle.dump(x_detected_all_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PATH_TO_SAVE + '/' + 'y_detected_all_all.pickle', 'wb') as handle:
        pickle.dump(y_detected_all_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PATH_TO_SAVE + '/' + 'code_detected_all_all.pickle', 'wb') as handle:
        pickle.dump(code_detected_all_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PATH_TO_SAVE + '/' + 'for_localization.pickle', 'wb') as handle:
        pickle.dump(for_localization, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    if not is_localized:
        if close_to_target_counter >= CLOSE_TO_TARGET_SKEEP_THRESHOLD:
            close_to_target = False
            close_to_target_counter = 0
        else:
            close_to_target_counter += 1
        
    
    print(' ')
    print('is_localized =', is_localized)
    print('current_solution =', current_solution)
    
    #if frame_index % 5 == 0:
    #    cv2.imwrite('./tt0038_tmp' + '/' + str(frame_index).zfill(4) + '.png', frame)
    
    frame_index += 1
    
    time_2 = time.time()
    time_detect = time_2 - time_1
    if time_detect < APHTER_PHOTO_PAUSE:
        time.sleep(APHTER_PHOTO_PAUSE - time_detect)
    
    stop_flag = True
    set_wheeles(0, 0)
    time.sleep(PRE_PHOTO_PAUSE)
    
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    
#time_2 = time.time()
#print((time_2 - time_1) / n_frames) # 0.3109789037704468
