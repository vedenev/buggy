import wiringpi


#pin  = 1 # gpio pin 12 = wiringpi no. 1 (BCM 18)
pin = 26



wiringpi.wiringPiSetup()
wiringpi.pinMode(pin, 2)     # PWM mode
wiringpi.pinMode(10, 1)
wiringpi.pinMode(11, 1)
wiringpi.pinMode(12, 1)
wiringpi.pinMode(13, 1)


wiringpi.pwmWrite(pin, 500)    # OFF

       # Set pin to 1 ( OUTPUT )
wiringpi.digitalWrite(13, 0) 
wiringpi.digitalWrite(12, 0) 


