import wiringpi
import numpy as np
import time


#https://github.com/WiringPi/WiringPi-Python/blob/master/examples/callback.py

MEASURE_PERIOD = 0.3

#LOOPBACK_COEFFICIENT = 0.5 * 2.28
#LOOPBACK_COEFFICIENT_INTEGRAL = 2.0 * 2.28




pin_PWM_right = 1
pin_forward_right = 11
pin_backward_right = 10

pin_PWM_left = 23
pin_forward_left = 13
pin_backward_left = 12

DIGITAL_OUT_MODE = 1
PWM_MODE = 2

#pin_encoder_right = 11
#pin_encoder_left = 13

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

def correct_value(value):
    if value > 1023:
        value = 1023
    if value < -1023:
        value = -1023
    return value

def int_round(inp):
    return int(np.round(inp))

def set_wheeles(right_value, left_value):
    right_value = int_round(right_value)
    left_value = int_round(left_value)
    right_value = correct_value(right_value)
    left_value = correct_value(left_value)
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

kp_min = -1.0
ki_min = -1.0
kd_min = -1.0
res_min = 1.0e7

for try_index in range(100):

    kp = 0.38 * (1 + 0.1 * (2.0 * np.random.rand() - 1.0))
    ki = 2.9 * (1 + 0.1 * (2.0 * np.random.rand() - 1.0))
    kd = 0.42 * (1 + 0.1 * (2.0 * np.random.rand() - 1.0))

    set_wheeles(0.0, 0.0)

    time.sleep(0.5)

    right_value = 0.0
    left_value = 0.0

    right_value_old = 0.0
    left_value_old = 0.0

    error_right_old = 0.0
    error_left_old = 0.0

    error_right_old2 = 0.0
    error_left_old2 = 0.0 

    time_tmp = 0.0
    time_index_all = np.arange(40)
    speed_right_all = np.zeros(time_index_all.size, np.float32)
    speed_left_all = np.zeros(time_index_all.size, np.float32)
    speed_right_target_all = np.zeros(time_index_all.size, np.float32)
    speed_left_target_all = np.zeros(time_index_all.size, np.float32)
    right_value_all = np.zeros(time_index_all.size, np.float32)
    left_value_all = np.zeros(time_index_all.size, np.float32)
    for time_index_index in range(time_index_all.size):
        time_index = time_index_all[time_index_index]
        
        #speed_right_target = 150.0 * np.sin(2 * np.pi * time_index / 60.0)
        #speed_left_target = - 150.0 * np.sin(2 * np.pi * time_index / 60.0)
        
        #speed_right_target = 50.0 * np.sign(np.sin(2 * np.pi * time_index / 40.0))
        #speed_left_target = - 50.0 * np.sign(np.cos(2 * np.pi * time_index / 40.0))
        
        speed_right_target = 100.0 * (time_index >= 10)
        speed_left_target = 100.0 * (time_index >= 10)
        
        
        set_wheeles(right_value, left_value)
        
        
        
        time_1 = time.time()
        counter_right_1 = counter_right
        counter_left_1 = counter_left
        
        time.sleep(MEASURE_PERIOD)
        time_tmp += MEASURE_PERIOD
        
        time_2 = time.time()
        counter_right_2 = counter_right
        counter_left_2 = counter_left
        
        time_diff = time_2 - time_1
        speed_right = (counter_right_2 - counter_right_1) / time_diff
        speed_left = (counter_left_2 - counter_left_1) / time_diff
        
        speed_right = np.sign(right_value) * speed_right
        speed_left = np.sign(left_value) * speed_left
        
        
        
        #right_value += int_round(LOOPBACK_COEFFICIENT * (speed_right_target - speed_right))
        #left_value += int_round(LOOPBACK_COEFFICIENT * (speed_left_target - speed_left))
        
        
        
        #right_value = int_round(LOOPBACK_COEFFICIENT * (speed_right_target - speed_right))
        #left_value = int_round(LOOPBACK_COEFFICIENT * (speed_left_target - speed_left))
       
       
       
        #right_value += int_round(LOOPBACK_COEFFICIENT_INTEGRAL * (speed_right_target - speed_right))
        #left_value += int_round(LOOPBACK_COEFFICIENT_INTEGRAL * (speed_left_target - speed_left))
        #right_value = correct_value(right_value)
        #left_value = correct_value(left_value)
        
        
        error_right = speed_right_target - speed_right
        error_left = (speed_left_target - speed_left)
        right_value = right_value_old + kp * (error_right - error_right_old) + ki * error_right + kd * (error_right - 2 * error_right_old + error_right_old2)
        left_value = left_value_old + kp * (error_left - error_left_old) + ki * error_left + kd * (error_left - 2 * error_left_old + error_left_old2)
        
        
        right_value_old = right_value
        left_value_old = left_value

        
        error_right_old2 = error_right_old
        error_left_old2 = error_left_old
        error_right_old = error_right
        error_left_old = error_left

        
        
        speed_right_all[time_index_index] = speed_right
        speed_left_all[time_index_index] = speed_left
        
        speed_right_target_all[time_index_index] = speed_right_target
        speed_left_target_all[time_index_index] = speed_left_target
        
        right_value_all[time_index_index] = right_value
        left_value_all[time_index_index] =  left_value
        
        print('right_value =', right_value, '  left_value =', left_value, ' speed_right =', speed_right, ' speed_left =', speed_left, ' speed_right_target =', speed_right_target, ' speed_left_target =', speed_left_target)
        

    set_wheeles(0, 0)
    
    res = np.mean(np.abs(speed_right_target_all - speed_right_all)) + np.mean(np.abs(speed_left_target_all - speed_left_all))
    
    if res < res_min:
        
        res_min = res
        kp_min = kp
        ki_min = ki
        kd_min = kd
        
        np.save('res_min.npy', res_min)
        np.save('kp_min.npy', kp_min)
        np.save('ki_min.npy', ki_min)
        np.save('kd_min.npy', kd_min)
        
        np.save('tt011_speed_right_all.npy', speed_right_all)
        np.save('tt011_speed_left_all.npy', speed_left_all)
        np.save('tt011_speed_right_target_all.npy', speed_right_target_all)
        np.save('tt011_speed_left_target_all.npy', speed_left_target_all)
        np.save('tt011_right_value_all.npy', right_value_all)
        np.save('tt011_left_value_all.npy', left_value_all)

