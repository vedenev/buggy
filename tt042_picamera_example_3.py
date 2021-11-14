import picamera
import picamera.array

WIDTH = 1280
HEIGHT = 960

theGodArray = None

# Inherit from PiRGBAnalysis
class MyAnalysisClass(picamera.array.PiRGBAnalysis):
    def analyse(self, array):
        print('array.shape =', array.shape)
        global theGodArray
        theGodArray = array

with picamera.PiCamera() as camera:
    with picamera.array.PiRGBAnalysis(camera) as output:
        camera.resolution = (HEIGHT, WIDTH)
        camera.framerate = 30
        output = MyAnalysisClass(camera)
        camera.start_recording(output, format='rgb')
        camera.wait_recording(5)
        camera.stop_recording()