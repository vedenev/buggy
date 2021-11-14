import cv2
import time
import numpy as np

SAVE_DIR = './photos_for_delay_test'

N_FRAMES = 50

WIDTH = 1280
HEIGHT = 960

frames = np.zeros((N_FRAMES, HEIGHT, WIDTH, 3), np.uint8)
dumb_matrix = 10.0 * np.random.rand(500, 500).astype(np.float32)

camera_capturer = cv2.VideoCapture(0)

if not (camera_capturer.isOpened()):
    print('Could not open video device')
    import sys
    sys.exit()

camera_capturer.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera_capturer.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

time_1 = time.time()
for frame_count in range(N_FRAMES):
    is_frame_read, frame = camera_capturer.read()
    if not is_frame_read:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #cv2.imwrite(SAVE_DIR + '/' + str(frame_count).zfill(3) + '.png', frame)
    
    for tries_index in range(50):
        dumb_matrix = np.sin(dumb_matrix)
        
    
    frames[frame_count, :, :, :] = frame
    time_2 = time.time()
    time_delta = time_2 - time_1
    print(f"{frame_count}  {time_delta :.2f}")
    
camera_capturer.release()

dumb_value = np.sum(dumb_matrix)
print("dumb_value =", dumb_value)

for frame_count in range(N_FRAMES):
    frame = frames[frame_count, :, :, :]
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    cv2.imwrite(SAVE_DIR + '/' + str(frame_count).zfill(3) + '.png', frame)