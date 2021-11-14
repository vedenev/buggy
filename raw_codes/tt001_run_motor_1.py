# https://projects.raspberrypi.org/en/projects/build-a-buggy/2
# pinout
from gpiozero import Robot
from gpiozero import Robot
robby = Robot(left=(7,8), right=(9,10))
robby.forward()