import cv2
import time

#SAVE_DIR = './photos_for_localization_test'
SAVE_DIR = './photos_for_angles'

WIDTH = 1280
HEIGHT = 960

camera_capturer = cv2.VideoCapture(0)

if not (camera_capturer.isOpened()):
    print('Could not open video device')
    import sys
    sys.exit()

camera_capturer.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
camera_capturer.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


frame_count = 0
while True:
    is_frame_read, frame = camera_capturer.read()
    if not is_frame_read:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imwrite(SAVE_DIR + '/' + str(frame_count).zfill(3) + '.png', frame)
    time.sleep(1.0)
    
    frame_count += 1

camera_capturer.release()
    