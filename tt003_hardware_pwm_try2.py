import pigpio
pi = pigpio.pi('pi', 8888)
pi.hardware_PWM(18, 800, 250000) # 800Hz 25% dutycycle
pi.hardware_PWM(18, 2000, 750000) # 2000Hz 75% dutycycle