# https://stackoverflow.com/questions/54269724/picamera-and-continuous-capture-with-python-and-raspberry-pi

from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import numpy as np
import cv2

WIDTH = 1296
HEIGHT = 736
N_FRAMES = 50

SAVE_DIR = './photos_for_delay_test'

frames = np.zeros((N_FRAMES, HEIGHT, WIDTH, 3), np.uint8)
frame_index = 0

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.resolution = (WIDTH, HEIGHT)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(WIDTH, HEIGHT))

# allow the camera to warmup
time.sleep(1.0)

# capture frames from the camera
time_1 = time.time()
for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
    image = frame.array
    
    frames[frame_index, :, :, :] = image
    time_2 = time.time()
    time_delta = time_2 - time_1
    print(f"{frame_index}  {time_delta :.2f}")
    


    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    
    if frame_index >= (N_FRAMES - 1):
        break
    
    frame_index += 1


for frame_count in range(N_FRAMES):
    frame = frames[frame_count, :, :, :]
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    cv2.imwrite(SAVE_DIR + '/' + str(frame_count).zfill(3) + '.png', frame)
    