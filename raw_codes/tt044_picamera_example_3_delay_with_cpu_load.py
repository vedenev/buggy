import picamera
import picamera.array
import numpy as np
import cv2
import time

#WIDTH = 1280
#HEIGHT = 960

WIDTH = 1296
HEIGHT = 736

N_FRAMES = 50


# https://raspberrypi.stackexchange.com/questions/32926/convert-the-frame-data-from-recording-into-a-numpy-array#32954

SAVE_DIR = './photos_for_delay_test'

frames = np.zeros((N_FRAMES, HEIGHT, WIDTH, 3), np.uint8)
frame_index = 0

dumb_matrix = 10.0 * np.random.rand(500, 500).astype(np.float32)

time_1 = time.time()

# Inherit from PiRGBAnalysis
class MyAnalysisClass(picamera.array.PiRGBAnalysis):
    def analyse(self, array):
        #print('array.shape =', array.shape)
        global frames
        global frame_index
        global dumb_matrix 
        
        for tries_index in range(50):
            dumb_matrix = np.sin(dumb_matrix)
        
        frames[frame_index, :, :, :] = array
        time_2 = time.time()
        time_delta = time_2 - time_1
        print(f"{frame_index}  {time_delta :.2f}")
        if frame_index < (N_FRAMES - 1):
            frame_index += 1
        

time.sleep(3.0)

with picamera.PiCamera() as camera:
    with picamera.array.PiRGBAnalysis(camera) as output:
        camera.resolution = (WIDTH, HEIGHT)
        camera.framerate = 1
        #print(dir(camera))
        #['AWB_MODES', 'CAMERA_CAPTURE_PORT', 'CAMERA_PREVIEW_PORT', 'CAMERA_VIDEO_PORT',
        #'CAPTURE_TIMEOUT', 'CLOCK_MODES', 'DEFAULT_ANNOTATE_SIZE', 'DRC_STRENGTHS', 'EXPOSURE_MODES',
        #'FLASH_MODES', 'IMAGE_EFFECTS', 'ISO', 'MAX_FRAMERATE', 'MAX_RESOLUTION', 'METER_MODES', 'RAW_FORMATS',
        #'STEREO_MODES', '_AWB_MODES_R', '_CLOCK_MODES_R', '_DRC_STRENGTHS_R', '_EXPOSURE_MODES_R', '_FLASH_MODES_R',
        #'_IMAGE_EFFECTS_R', '_METER_MODES_R', '_STEREO_MODES_R',
        #'__class__', '__delattr__', '__dir__', '__doc__', '__enter__', '__eq__', '__exit__', '__format__',
        #'__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__',
        #'__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__',
        #'__slots__', '__str__', '__subclasshook__', '_camera', '_camera_config', '_camera_exception', '_check_camera_open',
        #'_check_recording_stopped', '_configure_camera', '_configure_splitter', '_control_callback', '_disable_camera',
        #_enable_camera', '_encoders', '_encoders_lock', '_exif_tags', '_get_ISO', '_get_analog_gain', '_get_annotate_background',
        #'_get_annotate_foreground', '_get_annotate_frame_num', '_get_annotate_text', '_get_annotate_text_size', '_get_awb_gains',
        #'_get_awb_mode', '_get_brightness', '_get_clock_mode', '_get_color_effects', '_get_contrast', '_get_crop', '_get_digital_gain',
        #'_get_drc_strength', '_get_exposure_compensation', '_get_exposure_mode', '_get_exposure_speed', '_get_flash_mode', '_get_frame',
        #'_get_framerate', '_get_framerate_delta', '_get_framerate_range', '_get_hflip', '_get_image_denoise', '_get_image_effect',
        #'_get_image_effect_params', '_get_image_encoder', '_get_image_format', '_get_images_encoder', '_get_iso', '_get_meter_mode',
        #'_get_output_format', '_get_overlays', '_get_ports', '_get_preview', '_get_preview_alpha', '_get_preview_fullscreen',
        #'_get_preview_layer', '_get_preview_window', '_get_raw_format', '_get_resolution', '_get_rotation', '_get_saturation',
        #'_get_sensor_mode', '_get_sharpness', '_get_shutter_speed', '_get_still_stats', '_get_timestamp', '_get_vflip', '_get_video_denoise',
        #'_get_video_encoder', '_get_video_format', '_get_video_stabilization', '_get_zoom', '_image_effect_params', '_init_camera',
        #'_init_defaults', '_init_led', '_init_preview', '_init_splitter', '_led_pin', '_overlays', '_preview', '_preview_alpha',
        #'_preview_fullscreen', '_preview_layer', '_preview_window', '_raw_format', '_revision', '_set_ISO', '_set_annotate_background',
        #'_set_annotate_foreground', '_set_annotate_frame_num', '_set_annotate_text', '_set_annotate_text_size', '_set_awb_gains',
        #'_set_awb_mode', '_set_brightness', '_set_clock_mode', '_set_color_effects', '_set_contrast', '_set_crop', '_set_drc_strength',
        #'_set_exposure_compensation', '_set_exposure_mode', '_set_flash_mode', '_set_framerate', '_set_framerate_delta',
        #'_set_framerate_range', '_set_hflip', '_set_image_denoise', '_set_image_effect', '_set_image_effect_params', '_set_iso',
        #'_set_led', '_set_meter_mode', '_set_preview_alpha', '_set_preview_fullscreen', '_set_preview_layer', '_set_preview_window',
        #'_set_raw_format', '_set_resolution', '_set_rotation', '_set_saturation', '_set_sensor_mode', '_set_sharpness', '_set_shutter_speed',
        #'_set_still_stats', '_set_vflip', '_set_video_denoise', '_set_video_stabilization', '_set_zoom', '_splitter', '_splitter_connection',
        #'_start_capture', '_stop_capture', '_used_led', 'add_overlay', 'analog_gain', 'annotate_background', 'annotate_foreground',
        #'annotate_frame_num', 'annotate_text', 'annotate_text_size', 'awb_gains', 'awb_mode', 'brightness', 'capture', 'capture_continuous',
        #'capture_sequence', 'clock_mode', 'close', 'closed', 'color_effects', 'contrast', 'crop', 'digital_gain', 'drc_strength', 'exif_tags',
        #exposure_compensation', 'exposure_mode', 'exposure_speed', 'flash_mode', 'frame', 'framerate', 'framerate_delta', 'framerate_range',
        #'hflip', 'image_denoise', 'image_effect', 'image_effect_params', 'iso', 'led', 'meter_mode', 'overlays', 'preview', 'preview_alpha',
        #'preview_fullscreen', 'preview_layer', 'preview_window', 'previewing', 'raw_format', 'record_sequence', 'recording', 'remove_overlay',
        #'request_key_frame', 'resolution', 'revision', 'rotation', 'saturation', 'sensor_mode', 'sharpness', 'shutter_speed', 'split_recording',
        #'start_preview', 'start_recording', 'still_stats', 'stop_preview', 'stop_recording', 'timestamp', 'vflip', 'video_denoise',
        #'video_stabilization', 'wait_recording', 'zoom']
        output = MyAnalysisClass(camera)
        camera.start_recording(output, format='rgb')
        camera.wait_recording(20)
        camera.stop_recording()

dumb_value = np.sum(dumb_matrix)
print("dumb_value =", dumb_value)

for frame_count in range(N_FRAMES):
    frame = frames[frame_count, :, :, :]
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    cv2.imwrite(SAVE_DIR + '/' + str(frame_count).zfill(3) + '.png', frame)
