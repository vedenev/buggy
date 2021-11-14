import wiringpi
import numpy as np
import time

import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library



DELALAY = 5.0

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

pin_encoder_right = 11
counter_right = 0
def encoder_right_callback(channel):
    global counter_right
    print("er" + str(counter_right))
    counter_right += 1

pin_encoder_left = 13
counter_left = 0
def encoder_left_callback(channel):
    global counter_left
    print("el" + str(counter_left))
    counter_left += 1

GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
GPIO.setup(pin_encoder_right, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set
GPIO.add_event_detect(pin_encoder_right,GPIO.RISING,callback=encoder_right_callback) # Setup event on pin 10 rising edge

GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
GPIO.setup(pin_encoder_left, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set
GPIO.add_event_detect(pin_encoder_left,GPIO.RISING,callback=encoder_left_callback) # Setup event on pin 10 rising edge

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
N_steps = 1024
N_steps_step = 16
for time_index in range(0, N_steps, N_steps_step):
    
    
    right_value = time_index
    left_value = time_index
    
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
    



