import wiringpi
import numpy as np
import time


#https://github.com/WiringPi/WiringPi-Python/blob/master/examples/callback.py



DELALAY = 5.0

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

def set_wheeles(right_value, left_value):
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

set_wheeles(0, 0)

time.sleep(5.0)

time_tmp = 0.0
N_steps = 1023
N_steps_step = 8
time_index_all = np.arange(-N_steps, N_steps + 1, N_steps_step)
speed_right_all = np.zeros(time_index_all.size, np.float32)
speed_left_all = np.zeros(time_index_all.size, np.float32)
for time_index_index in range(time_index_all.size):
    time_index = time_index_all[time_index_index]
    
    
    right_value = int(time_index)
    left_value = int(time_index)
    
    set_wheeles(right_value, left_value)
    
    time.sleep(0.5)
    
    time_1 = time.time()
    counter_right_1 = counter_right
    counter_left_1 = counter_left
    
    time.sleep(DELALAY)
    time_tmp += DELALAY
    
    time_2 = time.time()
    counter_right_2 = counter_right
    counter_left_2 = counter_left
    
    time_diff = time_2 - time_1
    speed_right = (counter_right_2 - counter_right_1) / time_diff
    speed_left = (counter_left_2 - counter_left_1) / time_diff
    
    speed_right_all[time_index_index] = speed_right
    speed_left_all[time_index_index] = speed_left
    
    print('right_value =', right_value, '  left_value =', left_value, ' speed_right =', speed_right, ' speed_left =', speed_left)
    
np.save('time_index_all.npy', time_index_all)
np.save('speed_right_all.npy', speed_right_all)
np.save('speed_left_all.npy', speed_left_all)



