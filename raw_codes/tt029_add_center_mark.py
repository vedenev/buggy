import cv2

PATH = './028.png'
PATH_OUT = './028_central_mark.png'
LINE_WIDHT = 3

image = cv2.imread(PATH)
x = image.shape[1] // 2
y = image.shape[0] // 2
line_width_half = (LINE_WIDHT - 1) // 2

image[y - line_width_half: y + line_width_half, :, :] = 255
image[:, x - line_width_half: x + line_width_half, :] = 255

cv2.imwrite(PATH_OUT, image)

