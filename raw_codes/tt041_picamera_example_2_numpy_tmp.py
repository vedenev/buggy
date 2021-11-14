import picamera
import picamera.array

with picamera.PiCamera() as camera:
    with picamera.array.PiRGBArray(camera) as output:
        for tries in range(10):
            camera.capture(output, 'rgb')
            print('Captured %dx%d image' % (
                    output.array.shape[1], output.array.shape[0]))
