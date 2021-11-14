#https://raspberrypihq.com/use-a-push-button-with-raspberry-pi-gpio/

import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library

counter = 0

def encoder_callback(channel):
    global counter
    print("e" + str(counter))
    counter += 1




pin = 11



GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BOARD) # Use physical pin numbering
GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set
GPIO.add_event_detect(pin,GPIO.RISING,callback=encoder_callback) # Setup event on pin 10 rising edge