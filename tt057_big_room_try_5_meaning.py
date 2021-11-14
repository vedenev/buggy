from code_detector_2020_08_24 import detect_codes, y1_strip, y2_strip, WIDTH, HEIGHT
from localize_with_circles_2020_12_29 import localize_with_circles, codes_centres,  POSITIONS_CODES_REAL
from prepare_vector_field import vx, vy, RESOLUTION, TARGET_X, TARGET_Y, TARGET_TRAJECTORY, X, Y
import cv2
import numpy as np
import time
import glob
import os
import wiringpi
from picamera.array import PiRGBArray
from picamera import PiCamera
import pickle
import threading

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

#TRAJECTORY_LENGHT_MAX = 10
N_TRAJECTORIES_MEAN_MAX = 20

MANUAL_BALANCE_COEEFICIENT = -0.03;

JUST_DRAW = False;

PATH_TO_SAVE = 'tt053_saves'

MIN_TARGET_DISTANCE = 500.0 ** 2

UPDATE_WHEELS_VALUES_PERIOD = 0.03
PRE_PHOTO_PAUSE = 0.3
APHTER_PHOTO_PAUSE = 0.2
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
SMOOTH_CONTROL_BASE_VALUE = 405
SMOOTH_CONTROL_KP = 65.0

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

UPDATE_ODOMETRY_PERIOD = 5

counters_lock = threading.Lock()

if not os.path.exists(PATH_TO_SAVE):
    os.makedirs(PATH_TO_SAVE)
path_to_save_images = PATH_TO_SAVE + '/' + 'images'
if not os.path.exists(path_to_save_images):
    os.makedirs(path_to_save_images)
files = glob.glob(path_to_save_images + '/*')
for f in files:
    os.remove(f)

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


counter_right_old = 0
counter_left_old = 0

x_old = 0.0
y_old = 0.0
gamma_old = 0.0

x_old_old = 0.0
y_old_old = 0.0
gamma_old_old = 0.0

def update_odometry():
    global counter_right
    global counter_left
    global counter_right_old
    global counter_left_old
    global x_old
    global y_old
    global gamma_old
    with counters_lock:
        delta_right = counter_right - counter_right_old
        delta_left = counter_left - counter_left_old
        counter_right_old = counter_right
        counter_left_old = counter_left
        #N_holes = 20
        #r_wheel = 31 # mm
        #a_wheels = 118 # mm, half-distance between wheels
        # pi_r_N = np.pi * r_wheel / N_holes
        # pi_r_N_a = pi_r_N  / a_wheels
        alpha = pi_r_N_a * (delta_right - delta_left)
        alpha_half = alpha  / 2
        gamma_new = gamma_old + alpha
        amplitude_tmp = (delta_right + delta_left) * pi_r_N * np.sinc(alpha_half / np.pi)
        angle_tmp = gamma_old + alpha_half
        x_new = x_old + amplitude_tmp * np.cos(angle_tmp);
        y_new = y_old + amplitude_tmp * np.sin(angle_tmp);
        
        
        x_old = x_new
        y_old = y_new
        gamma_old = gamma_new

counter_right = 0
def encoder_right_callback():
    global counter_right
    global counter_left
    #print("er" + str(counter_right))
    counter_right += 1
    if (counter_right + counter_left) % UPDATE_ODOMETRY_PERIOD == 0:
        update_odometry()
    

counter_left = 0
def encoder_left_callback():
    global counter_right
    global counter_left
    #print("el" + str(counter_left))
    counter_left += 1
    if (counter_right + counter_left) % UPDATE_ODOMETRY_PERIOD == 0:
        update_odometry()



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

#wiringpi.wiringPiISR(pin_encoder_right, wiringpi.GPIO.INT_EDGE_BOTH, encoder_right_callback)
#wiringpi.wiringPiISR(pin_encoder_left, wiringpi.GPIO.INT_EDGE_BOTH, encoder_left_callback)

wiringpi.wiringPiISR(pin_encoder_right, wiringpi.GPIO.INT_EDGE_RISING, encoder_right_callback)
wiringpi.wiringPiISR(pin_encoder_left, wiringpi.GPIO.INT_EDGE_RISING, encoder_left_callback)

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

close_to_target = False

def correct_value(value):
    if value > 1023:
        value = 1023
    if value < -1023:
        value = -1023
    return value

def int_round(inp):
    return int(np.round(inp))

def set_wheeles(right_value_, left_value_):
    #print("right_value_ =", right_value_)
    #print("left_value_ =", left_value_)
    right_value_ = int_round(right_value_)
    left_value_ = int_round(left_value_)
    right_value_ = correct_value(right_value_)
    left_value_ = correct_value(left_value_)
    
    if right_value_ == 0:
        wiringpi.pwmWrite(pin_PWM_right, 0)
        wiringpi.digitalWrite(pin_forward_right, 0) 
        wiringpi.digitalWrite(pin_backward_right, 0)
    else:
        right_value_abs = abs(right_value_)
        wiringpi.pwmWrite(pin_PWM_right, right_value_abs)
        if right_value_ > 0:
            wiringpi.digitalWrite(pin_forward_right, 1) 
            wiringpi.digitalWrite(pin_backward_right, 0)
        else:
            wiringpi.digitalWrite(pin_forward_right, 0) 
            wiringpi.digitalWrite(pin_backward_right, 1)

    if left_value_ == 0:
        wiringpi.pwmWrite(pin_PWM_left, 0)
        wiringpi.digitalWrite(pin_forward_left, 0) 
        wiringpi.digitalWrite(pin_backward_left, 0)
    else:
        left_value_abs = abs(left_value_)
        wiringpi.pwmWrite(pin_PWM_left, left_value_abs)
        if left_value_ > 0:
            wiringpi.digitalWrite(pin_forward_left, 1) 
            wiringpi.digitalWrite(pin_backward_left, 0)
        else:
            wiringpi.digitalWrite(pin_forward_left, 0) 
            wiringpi.digitalWrite(pin_backward_left, 1)

close_to_target_counter = 0

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (WIDTH, HEIGHT)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(WIDTH, HEIGHT))

# allow the camera to warmup
time.sleep(1.0)



is_localized_all = np.zeros(0, np.bool)
is_localized_by_odometry_all = np.zeros(0, np.bool)
current_solution_all = np.zeros((0, 3), np.float32)
is_localized_tmp_all = np.zeros(0, np.bool)
current_solution_tmp_all = np.zeros((0, 3), np.float32)
x_detected_all_all = []
y_detected_all_all = []
code_detected_all_all = []
for_localization = []
counter_right_all = np.zeros(0, np.int32)
counter_left_all = np.zeros(0, np.int32)
right_value_all = np.zeros(0, np.int32)
left_value_all = np.zeros(0, np.int32)
frame_tmp_all = []

is_localized_by_odometry = False;
is_localized_once = False

trajectories = []
# trajectories[trajectory_no][point_no_in_time] = [x, y, gamma, cos(gamma), sin(gamma)]

print_no = 0
frame_index = 0
#n_frames = 100
#time_1 = time.time()
#for frame_index in range(n_frames):
for frame_0 in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame = frame_0.array
    
    #print(dir(frame_0))
    #['__class__', '__del__', '__delattr__', '__dict__', '__dir__', '__doc__', '__enter__', '__eq__', '__exit__', '__format__',
    #'__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__iter__', '__le__',
    #'__lt__', '__module__', '__ne__', '__new__', '__next__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__',
    #'__setstate__', '__sizeof__', '__str__', '__subclasshook__', '_checkClosed', '_checkReadable', '_checkSeekable', '_checkWritable',
    #'array', 'camera', 'close', 'closed', 'detach', 'fileno', 'flush', 'getbuffer', 'getvalue', 'isatty', 'read', 'read1', 'readable',
    #'readinto', 'readinto1', 'readline', 'readlines', 'seek', 'seekable', 'size', 'tell', 'truncate', 'writable', 'write', 'writelines']
    
    #print(dir(frame_0))
    

    #stop_flag = False
    ##set_wheeles(right_value_global, left_value_global)
    
    if close_to_target:
        set_wheeles(0, 0)
    else:
        set_wheeles(right_value, left_value)
    
    
    ## for debug:
    #frame = cv2.imread('./photos_for_localization_test/028.png')
    
    time_1 = time.time()
    x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all, frame_tmp = detect_codes(np.copy(frame))
    x_detected_all_all.append(x_detected_all)
    y_detected_all_all.append(y_detected_all)
    
    for_localization_tmp = [x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all]
    for_localization.append(for_localization_tmp)
    #frame_tmp_all.append(frame_tmp)
    cv2.imwrite(path_to_save_images + '/' + str(frame_index).zfill(3) + '.jpg', frame_tmp)
    
    is_localized_tmp, current_solution_tmp, code_detected_all, angles_detected_pairs_all, detected_to_real_indexing = localize_with_circles(x_detected_all, y_detected_all, code_detected_all, code_size_all, x_detected_pairs_all, y_detected_pairs_all)
    #if not is_localized:
    #    current_solution  = np.zeros((3, 1), np.float32)
    #    current_solution[0, 0] = 0.0
    #    current_solution[1, 0] = 2000.0
    #    current_solution[2, 0] = -np.pi/2
    
    
    is_localized_tmp_all = np.append(is_localized_tmp_all, is_localized_tmp)
    current_solution_tmp_all = np.concatenate((current_solution_tmp_all, current_solution_tmp.reshape(1, 3)), axis=0)
    
    print('is_localized_tmp =', is_localized_tmp)
    
    
    code_detected_all_all.append(code_detected_all) 
    
    update_odometry()
    
    x_old_tmp = x_old
    y_old_tmp = y_old
    gamma_old_tmp = gamma_old
    
    
    delta_x_old_tmp = x_old_tmp - x_old_old
    delta_y_old_tmp = y_old_tmp - y_old_old
    cos_gamma_old_tmp = np.cos(gamma_old_old)
    sin_gamma_old_tmp = np.sin(gamma_old_old)
    # dx and dy in old cooridinates system:
    delta_x_old = delta_x_old_tmp * cos_gamma_old_tmp + delta_y_old_tmp * sin_gamma_old_tmp
    delta_y_old = delta_x_old_tmp * (-sin_gamma_old_tmp) + delta_y_old_tmp * cos_gamma_old_tmp 
    delta_gamma_old = gamma_old_tmp - gamma_old_old
    
    # return of gamma to new system?:
    #  = 1 * cos_gamma + 0 * (-sin_gamma)
    #  = 1 * sin_gamma + 0 * sin_gamma
    # gamma vector in old coodinates system:
    # 1  = cos_gamma_old_tmp * cos_gamma_old_tmp + sin_gamma_old_tmp * sin_gamma_old_tmp
    # 0  = cos_gamma_old_tmp * (-sin_gamma_old_tmp) + sin_gamma_old_tmp * cos_gamma_old_tmp 
    
    trajectories_new = []
    for trajectory_index in range(len(trajectories)):
        trajectory = trajectories[trajectory_index]
        #if len(trajectory) < TRAJECTORY_LENGHT_MAX:
        if True:
            point_last = trajectory[-1]
            x_tmp_7, y_tmp_7, gamma_tmp_7, cos_tmp_7, sin_tmp_7 = point_last
            dx_tmp_7 = delta_x_old * cos_tmp_7 - delta_y_old * sin_tmp_7
            dy_tmp_7 = delta_x_old * sin_tmp_7 + delta_y_old * cos_tmp_7
            x_tmp_8 = x_tmp_7 + dx_tmp_7
            y_tmp_8 = y_tmp_7 + dy_tmp_7
            gamma_tmp_8 = gamma_tmp_7 + delta_gamma_old
            cos_tmp_8 = np.cos(gamma_tmp_8)
            sin_tmp_8 = np.sin(gamma_tmp_8)
            point_new = [x_tmp_8, y_tmp_8, gamma_tmp_8, cos_tmp_8, sin_tmp_8]
            trajectory.append(point_new)
            trajectories_new.append(trajectory)
            
    trajectories = trajectories_new
    
    x_old_old = x_old_tmp
    y_old_old = y_old_tmp
    gamma_old_old = gamma_old_tmp
    
    if is_localized_tmp:
        trajectory_new = [[current_solution_tmp[0, 0], current_solution_tmp[1, 0], current_solution_tmp[2, 0], np.cos(current_solution_tmp[2, 0]), np.sin(current_solution_tmp[2, 0])]]
        trajectories.append(trajectory_new)
    
    #is_localized_by_odometry = False;
    #if is_localized_tmp:
    #    is_localized_once = True
    #    #with counters_lock:
    #    #    counter_right_old = counter_right
    #    #    counter_left_old = counter_left
    #    #    x_old = current_solution_tmp[0, 0]
    #    #    y_old = current_solution_tmp[1, 0]
    #    #    gamma_old = current_solution_tmp[2, 0]
    #else:
    #    if is_localized_once:
    #        is_localized_by_odometry = True;
    #        is_localized_tmp = True;
    #        #delta_right = counter_right_tmp - counter_right_old
    #        #delta_left = counter_left_tmp - counter_left_old
    #        ##N_holes = 20
    #        ##r_wheel = 31 # mm
    #        ##a_wheels = 118 # mm, half-distance between wheels
    #        ## pi_r_N = np.pi * r_wheel / N_holes
    #        ## pi_r_N_a = pi_r_N  / a_wheels
    #        #alpha = pi_r_N_a * (delta_right - delta_left)
    #        #gamma_new = gamma_old + alpha
    #        #amplitude_tmp = (delta_right + delta_left) * pi_r_N * np.sinc(alpha / (2 * np.pi))
    #        #angle_tmp = gamma_old + alpha / 2
    #        #x_new = x_old + amplitude_tmp * np.cos(angle_tmp);
    #        #y_new = y_old + amplitude_tmp * np.sin(angle_tmp);
    #        #current_solution_tmp[0, 0] = x_new
    #        #current_solution_tmp[1, 0] = y_new
    #        #current_solution_tmp[2, 0] = gamma_new
    #        
    #        #update_odometry()
    #        
    #        current_solution_tmp[0, 0] = x_old
    #        current_solution_tmp[1, 0] = y_old
    #        current_solution_tmp[2, 0] = gamma_old
    
    # todo: try to fix this, unprecise motion to 8-code:
    #print_no = 97
    #is_localized = True
    #current_solution = [[2.6054502e+03]
    # [3.6780364e+03]
    # [1.4810907e+00]]
    #close_to_target = False
    #is_localized_by_odometry  = True
    #within_map = True
    #is_localized_and_within_map = True
    #is_localized_tmp = False
    #trajectories[-5:] = [[[2312.031, 3399.859, 2.1391063, -0.5382084, 0.84281176], [2279.073982024516, 3453.894499886004, 2.0978395904895626, -0.5029800421742006, 0.864298025668484], [2244.0008934820426, 3517.9571033844754, 2.056572907328002, -0.46689524014338407, 0.8843126340449127], [2206.1414802895597, 3596.975932380816, 1.9740395410048812, -0.39240348714125955, 0.91979318505813], [2179.0606119813447, 3664.805468914477, 1.9327728578433208, -0.3541233730905208, 0.9351987150499042], [2154.738067669703, 3728.471116530098, 1.9327728578433208, -0.3541233730905208, 0.9351987150499042], [2133.550516198485, 3788.1125506569683, 1.8915061746817603, -0.3152402940626639, 0.9490118845405916], [2120.0249733456126, 3829.786405854747, 1.9327728578433208, -0.3541233730905208, 0.9351987150499042], [2098.4587955067386, 3889.2933341454327, 1.9740395410048812, -0.39240348714125955, 0.91979318505813], [2084.807642173945, 3920.523587571396, 2.0153062241664417, -0.4300154567609198, 0.9028215255224575], [2082.804986348736, 3924.961799986516, 1.9740395410048812, -0.39240348714125955, 0.91979318505813], [2082.804986348736, 3924.961799986516, 1.9740395410048812, -0.39240348714125955, 0.91979318505813], [2082.804986348736, 3924.961799986516, 1.9740395410048812, -0.39240348714125955, 0.91979318505813], [2082.804986348736, 3924.961799986516, 1.9740395410048812, -0.39240348714125955, 0.91979318505813], [2082.804986348736, 3924.961799986516, 1.9740395410048812, -0.39240348714125955, 0.91979318505813]], [[2290.6113, 3448.52, 2.0902038, -0.49636582, 0.86811346], [2256.0284280734963, 3512.848564713292, 2.048937078892883, -0.46012923490280117, 0.8878519511651494], [2218.7734869467745, 3592.154175264353, 1.966403712569762, -0.3853687327759873, 0.9227626671027765], [2192.21133779231, 3660.1885172321718, 1.9251370294082015, -0.34697210185676847, 0.9378754504373682], [2168.3756377891295, 3724.038029786823, 1.9251370294082015, -0.34697210185676847, 0.9378754504373682], [2147.6441113273836, 3783.839508134948, 1.883870346246641, -0.3079846824119834, 0.951391315600258], [2134.437174098141, 3825.615426144409, 1.9251370294082015, -0.34697210185676847, 0.9378754504373682], [2113.3260052532482, 3885.2852936754603, 1.966403712569762, -0.3853687327759873, 0.9227626671027765], [2099.913716429544, 3916.618873504054, 2.0076703957313224, -0.4231091973227004, 0.9060786980946745], [2097.9450080865877, 3921.0722483205664, 1.966403712569762, -0.3853687327759873, 0.9227626671027765], [2097.9450080865877, 3921.0722483205664, 1.966403712569762, -0.3853687327759873, 0.9227626671027765], [2097.9450080865877, 3921.0722483205664, 1.966403712569762, -0.3853687327759873, 0.9227626671027765], [2097.9450080865877, 3921.0722483205664, 1.966403712569762, -0.3853687327759873, 0.9227626671027765], [2097.9450080865877, 3921.0722483205664, 1.966403712569762, -0.3853687327759873, 0.9227626671027765]], [[2287.4858, 3541.9607, 2.0794187, -0.48697442, 0.87341624], [2247.8312199362017, 3620.094052330019, 1.9968852928870842, -0.4133126273615155, 0.9105891895160633], [2219.207936490174, 3687.28725957202, 1.9556186097255237, -0.37539442291630504, 0.9268651613062896], [2193.4373761466145, 3750.3806749787577, 1.9556186097255237, -0.37539442291630504, 0.9268651613062896], [2170.8929187439394, 3809.522542065233, 1.9143519265639632, -0.33683703533091214, 0.9415629621164385], [2156.4189176806667, 3850.876548034797, 1.9556186097255237, -0.37539442291630504, 0.9268651613062896], [2133.4990053336237, 3909.8752951331435, 1.9968852928870842, -0.4133126273615155, 0.9105891895160633], [2119.1379977495685, 3940.7855552021892, 2.0381519760486446, -0.4505270854351586, 0.8927627598020099], [2117.03447904331, 3945.1768612548794, 1.9968852928870842, -0.4133126273615155, 0.9105891895160633], [2117.03447904331, 3945.1768612548794, 1.9968852928870842, -0.4133126273615155, 0.9105891895160633], [2117.03447904331, 3945.1768612548794, 1.9968852928870842, -0.4133126273615155, 0.9105891895160633], [2117.03447904331, 3945.1768612548794, 1.9968852928870842, -0.4133126273615155, 0.9105891895160633], [2117.03447904331, 3945.1768612548794, 1.9968852928870842, -0.4133126273615155, 0.9105891895160633]], [[2225.3354, 3572.7231, 1.9892445, -0.40634298, 0.9137206], [2197.2264073975985, 3640.13309229587, 1.9479777778980099, -0.36830151285890883, 0.9297064029175227], [2171.9386808084896, 3703.4215725435633, 1.9479777778980099, -0.36830151285890883, 0.9297064029175227], [2149.846770164667, 3762.7339699505196, 1.9067110947364494, -0.32963294847126473, 0.944109167036387], [2135.6891675425736, 3804.19736108662, 1.9479777778980099, -0.36830151285890883, 0.9297064029175227], [2113.2207193691734, 3863.3695114434386, 1.9892444610595703, -0.4063429711864941, 0.9137206300436321], [2099.0963087983, 3894.3885981867766, 2.0305111442211308, -0.44369255035856, 0.8961790673500005], [2097.0264044004516, 3898.7958485290806, 1.9892444610595703, -0.4063429711864941, 0.9137206300436321], [2097.0264044004516, 3898.7958485290806, 1.9892444610595703, -0.4063429711864941, 0.9137206300436321], [2097.0264044004516, 3898.7958485290806, 1.9892444610595703, -0.4063429711864941, 0.9137206300436321], [2097.0264044004516, 3898.7958485290806, 1.9892444610595703, -0.4063429711864941, 0.9137206300436321], [2097.0264044004516, 3898.7958485290806, 1.9892444610595703, -0.4063429711864941, 0.9137206300436321]], [[2211.957, 3662.816, 1.9834054, -0.40100077, 0.91607773], [2184.443483604163, 3725.1689906927395, 1.983405351638794, -0.4010007596131842, 0.9160777209329181], [2160.264580632086, 3783.6616710307294, 1.9421386684772335, -0.36286660765273465, 0.9318410943130788], [2144.647221692354, 3824.597579671651, 1.983405351638794, -0.4010007596131842, 0.9160777209329181], [2120.0969850379706, 3882.93676394247, 2.0246720348003544, -0.43845212858833027, 0.8987545443202846], [2104.882736288596, 3913.4360975640016, 2.065938717961915, -0.47515694622636906, 0.8799010606044474], [2102.658025210127, 3917.7672660452904, 2.0246720348003544, -0.43845212858833027, 0.8987545443202846], [2102.658025210127, 3917.7672660452904, 2.0246720348003544, -0.43845212858833027, 0.8987545443202846], [2102.658025210127, 3917.7672660452904, 2.0246720348003544, -0.43845212858833027, 0.8987545443202846], [2102.658025210127, 3917.7672660452904, 2.0246720348003544, -0.43845212858833027, 0.8987545443202846], [2102.658025210127, 3917.7672660452904, 2.0246720348003544, -0.43845212858833027, 0.8987545443202846]]]
    #trajectories_lengths = [79 78 77 76 75 74 72 71 70 69 68 63 62 61 60 59 58 57 56 55 54 53 52 51
    # 50 48 47 46 45 44 43 42 41 40 39 38 37 36 35 34 26 25 24 23 22 21 20 17
    # 16 15 14 13 12 11]
    #trajectories_selected = []
    #Traceback (most recent call last):
    #  File "tt057_big_room_try_5_meaning.py", line 525, in <module>
    #    x_mean = x_sum / weights_sum
    #ZeroDivisionError: float division by zero


    print('trajectories[-5:] =', trajectories[-5:])
    
    # last points of trajectories meaning:
    if len(trajectories) > 0:
        trajectories_lengths = np.zeros(len(trajectories), np.int64)
        for trajectory_index in range(len(trajectories)):
            trajectory = trajectories[trajectory_index]
            len_tmp = len(trajectory) 
            trajectories_lengths[trajectory_index] = len_tmp
        
        print('trajectories_lengths =', trajectories_lengths)
        
        #indexes_short = np.nonzero(trajectories_lengths <= TRAJECTORY_LENGHT_MAX)[0]
        #if indexes_short.size > 0:
        #    indexes_selected = indexes_short
        #else:
        #    indexes_selected = np.asarray([np.argmin(trajectories_lengths)])
        
        indexes_selected = np.argsort(trajectories_lengths)
        if indexes_selected.size > N_TRAJECTORIES_MEAN_MAX:
            indexes_selected = indexes_selected[0: N_TRAJECTORIES_MEAN_MAX]

        trajectories_selected = []
        for index_index in range(indexes_selected.size):
            index = indexes_selected[index_index]
            trajectory = trajectories[index]
            trajectories_selected.append(trajectory)
            
            
        print('trajectories_selected =', trajectories_selected)        
            
        x_sum = 0.0
        y_sum = 0.0
        cos_sum = 0.0
        sin_sum = 0.0
        weights_sum = 0.0
        for trajectory_index in range(len(trajectories_selected)):
            trajectory = trajectories_selected[trajectory_index]
            point_last = trajectory[-1]
            x_tmp_7, y_tmp_7, gamma_tmp_7, cos_tmp_7, sin_tmp_7 = point_last
            weight = 1.0 / len(trajectory)
            x_sum += x_tmp_7 * weight
            y_sum += y_tmp_7 * weight
            cos_sum += cos_tmp_7 * weight
            sin_sum += sin_tmp_7 * weight
            weights_sum += weight
        x_mean = x_sum / weights_sum
        y_mean = y_sum / weights_sum
        cos_mean = cos_sum / weights_sum
        sin_mean = sin_sum / weights_sum
        gamma_mean = np.arctan2(sin_mean, cos_mean)
        
        is_localized = True
        is_localized_by_odometry = not is_localized_tmp
        
        current_solution = np.zeros((3, 1), np.float32)
        current_solution[0, 0] = x_mean
        current_solution[1, 0] = y_mean
        current_solution[2, 0] = gamma_mean
        print("weights_sum =", weights_sum)
    else:
        is_localized = False
        is_localized_by_odometry = False
        current_solution = np.copy(current_solution_tmp)
    
    
       
        
    
    is_localized_all = np.append(is_localized_all, is_localized)
    is_localized_by_odometry_all = np.append(is_localized_by_odometry_all, is_localized_by_odometry)
    current_solution_all = np.concatenate((current_solution_all, current_solution.reshape(1, 3)), axis=0)
    counter_right_all = np.append(counter_right_all, counter_right)
    counter_left_all = np.append(counter_left_all, counter_left)
    right_value_all = np.append(right_value_all, right_value)
    left_value_all = np.append(left_value_all, left_value)
    
    np.save(PATH_TO_SAVE + '/' + 'is_localized_all' + '.npy', is_localized_all)
    np.save(PATH_TO_SAVE + '/' + 'is_localized_by_odometry_all' + '.npy', is_localized_by_odometry_all)
    np.save(PATH_TO_SAVE + '/' + 'current_solution_all' + '.npy', current_solution_all)
    np.save(PATH_TO_SAVE + '/' + 'is_localized_tmp_all' + '.npy', is_localized_tmp_all)
    np.save(PATH_TO_SAVE + '/' + 'current_solution_tmp_all' + '.npy', current_solution_tmp_all)
    np.save(PATH_TO_SAVE + '/' + 'counter_right_all' + '.npy', counter_right_all)
    np.save(PATH_TO_SAVE + '/' + 'counter_left_all' + '.npy', counter_left_all)
    np.save(PATH_TO_SAVE + '/' + 'right_value_all' + '.npy', right_value_all)
    np.save(PATH_TO_SAVE + '/' + 'left_value_all' + '.npy', left_value_all)
    with open(PATH_TO_SAVE + '/' + 'x_detected_all_all.pickle', 'wb') as handle:
        pickle.dump(x_detected_all_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PATH_TO_SAVE + '/' + 'y_detected_all_all.pickle', 'wb') as handle:
        pickle.dump(y_detected_all_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PATH_TO_SAVE + '/' + 'code_detected_all_all.pickle', 'wb') as handle:
        pickle.dump(code_detected_all_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PATH_TO_SAVE + '/' + 'for_localization.pickle', 'wb') as handle:
        pickle.dump(for_localization, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open(PATH_TO_SAVE + '/' + 'frame_tmp_all.pickle', 'wb') as handle:
    #    pickle.dump(frame_tmp_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #if not is_localized:
    #    if close_to_target_counter >= CLOSE_TO_TARGET_SKEEP_THRESHOLD:
    #        close_to_target = False
    #        close_to_target_counter = 0
    #    else:
    #        close_to_target_counter += 1
        
    
    print(' ')
    print('print_no =', print_no)
    print('is_localized =', is_localized)
    print('current_solution =', current_solution)
    print('close_to_target =', close_to_target)
    print('is_localized_by_odometry  =', is_localized_by_odometry )
    print_no += 1
    
    #if frame_index % 5 == 0:
    #    cv2.imwrite('./tt0038_tmp' + '/' + str(frame_index).zfill(4) + '.png', frame)
    
    frame_index += 1
    
    time_2 = time.time()
    time_detect = time_2 - time_1
    if time_detect < APHTER_PHOTO_PAUSE:
        time.sleep(APHTER_PHOTO_PAUSE - time_detect)
    
    set_wheeles(0, 0)
    
    time_1 = time.time()
    
    x = current_solution[0, 0]
    y = current_solution[1, 0]
    angle = current_solution[2, 0]
    if is_localized:
        target_distance = (TARGET_X - x)**2 +  (TARGET_Y - y)**2
        if target_distance < MIN_TARGET_DISTANCE:
            close_to_target = True
            close_to_target_counter = 0
        else:
            if close_to_target_counter >= CLOSE_TO_TARGET_SKEEP_THRESHOLD:
                close_to_target = False
                close_to_target_counter = 0
            else:
                close_to_target_counter += 1
    

    
    right_value = 0
    left_value = 0
    is_localized_and_within_map = False
    
    
    if not close_to_target:
        
        if is_localized:
            x_px = int_round(x * RESOLUTION)
            y_px = int_round(y * RESOLUTION)
            within_map = False
            if 0 <= x_px <= vx.shape[1] - 1 and 0 <= y_px <= vx.shape[0] - 1:
                vx_tmp = vx[y_px, x_px]
                vy_tmp = vy[y_px, x_px]
                v_norm_tmp = np.sqrt(vx_tmp**2 + vy_tmp**2)
                if v_norm_tmp > 0:
                    vx_tmp = vx_tmp / v_norm_tmp
                    vy_tmp = vy_tmp / v_norm_tmp
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
                        right_value = MINIMAL_MOVABLE_VALUE_RIGHT
                        left_value = 0
                    else:
                        right_value = 0
                        left_value = MINIMAL_MOVABLE_VALUE_LEFT
                else:
                    # smooth control
                    #SMOOTH_CONTROL_BASE_VALUE = 370
                    #SMOOTH_CONTROL_KP = 35.0
                    if pseudo_product_tmp > 1.0:
                        pseudo_product_tmp = 1.0
                    if pseudo_product_tmp < -1.0:
                        pseudo_product_tmp = -1.0
                    error = np.arcsin(pseudo_product_tmp)
                    right_value = int_round((1.0 + MANUAL_BALANCE_COEEFICIENT) * (SMOOTH_CONTROL_BASE_VALUE + SMOOTH_CONTROL_KP * error))
                    if right_value < 0:
                        right_value = 0
                    left_value = int_round((1.0 - MANUAL_BALANCE_COEEFICIENT) * (SMOOTH_CONTROL_BASE_VALUE - SMOOTH_CONTROL_KP * error))
                    if left_value < 0:
                        left_value = 0
    
    
    print('is_localized_and_within_map =', is_localized_and_within_map)
    if not is_localized_and_within_map:
        
        right_value = right_value_global
        left_value = left_value_global
     
    #set_wheeles(right_value, left_value)
        
    
    right_value_global = right_value
    left_value_global = left_value
    
    
    #stop_flag = True
    #time.sleep(PRE_PHOTO_PAUSE)
    
    time_2 = time.time()
    time_tmp = time_2 - time_1
    if time_tmp < PRE_PHOTO_PAUSE:
        time.sleep(PRE_PHOTO_PAUSE - time_tmp)
    
    
    
    
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    
#time_2 = time.time()
#print((time_2 - time_1) / n_frames) # 0.3109789037704468
