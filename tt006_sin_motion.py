import wiringpi
import numpy as np
import time


DELALAY = 0.1

pin_PWM_right = 1
pin_forward_right = 11
pin_backward_right = 10

pin_PWM_left = 23
pin_forward_left = 13
pin_backward_left = 12

DIGITAL_OUT_MODE = 1
PWM_MODE = 2

right_value = 0
left_value = 0

wiringpi.wiringPiSetup()

# set PWM mode:
wiringpi.pinMode(pin_PWM_right, PWM_MODE)
wiringpi.pinMode(pin_PWM_left, PWM_MODE)

# set digital output mode:
wiringpi.pinMode(pin_forward_right, DIGITAL_OUT_MODE)
wiringpi.pinMode(pin_backward_right, DIGITAL_OUT_MODE)
wiringpi.pinMode(pin_forward_left, DIGITAL_OUT_MODE)
wiringpi.pinMode(pin_backward_left, DIGITAL_OUT_MODE)

time_tmp = 0.0
for time_index in range(10000):
    
    diff_tmp = 10 * np.sin(2 * np.pi * time_tmp / 3.0)
    
    right_value = int(np.round(320 + diff_tmp))
    left_value = int(np.round(320 - diff_tmp))
    
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

    time.sleep(DELALAY)
    time_tmp += DELALAY
    
    print('right_value =', right_value, '  left_value =', left_value)
    


