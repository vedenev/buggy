# https://www.raspberrypi.org/forums/viewtopic.php?p=273886#p273886
# https://raspberrypi.stackexchange.com/questions/40105/access-gpio-pins-without-root-no-access-to-dev-mem-try-running-as-root

#sudo groupadd gpio
#sudo usermod -a -G gpio pi
#sudo grep gpio /etc/group
#sudo chown root.gpio /dev/gpiomem
#sudo chmod g+rw /dev/gpiomem

import wiringpi


pin  = 1 # gpio pin 12 = wiringpi no. 1 (BCM 18)

# Initialize PWM output for LED
wiringpi.wiringPiSetup()
wiringpi.pinMode(pin, 2)     # PWM mode
wiringpi.pwmWrite(pin, 0)    # OFF