import numpy as np
import matplotlib.pyplot as plt

WIDTH = 1280
HEIGHT = 960

#https://docs.opencv.org/2.4.1/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
mtx = np.array([[617.8249050804442, 0.0, 673.0536941293645], [0.0, 619.3492046143635, 497.9661474464693], [0.0, 0.0, 1.0]])
dist = np.array([[-0.3123562037471547, 0.1018281655721802, 0.00031297833728767365, 0.0007424882126541622, -0.015160446251882953]])

k1 = dist[0, 0]
k2 = dist[0, 1]
p1 = dist[0, 2]
p2 = dist[0, 3]
k3 = dist[0, 4]

fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]


alpha_x_0 = np.linspace(-np.pi / 3, np.pi / 3, 57)
alpha_y_0 = np.linspace(-np.pi / 4, np.pi / 4, 5)

alpha_x, alpha_y = np.meshgrid(alpha_x_0, alpha_y_0)
   
#xs = x / z
# tan(alpha) = x / z = xs

xs = np.tan(alpha_x)
ys = np.tan(alpha_y)

r2 = xs ** 2 + ys ** 2
r4 = r2 ** 2
r6 = r4 * r2
multimplier_tmp = (1 + k1 * r2 + k2 * r4 + k3 * r6) 
xss = xs * multimplier_tmp + 2 * p1 * xs * ys + p2 * (r2 + 2 * xs**2)
yss = ys * multimplier_tmp + p1 * (r2 + 2 * ys**2) + 2 * p2 * xs * ys

u = fx * xss + cx
v = fy * yss + cy

PIXELS_TO_ANGLE = np.full((HEIGHT, WIDTH),np.nan, np.float32)

alpha_x_flat =alpha_x.flatten();
u_flat = np.round(u.flatten()).astype(np.int64);
v_flat = np.round(v.flatten()).astype(np.int64);
                  
indexes = np.nonzero((0 <= u_flat) & (u_flat < WIDTH) & (0 <= v_flat) & (v_flat < HEIGHT))[0]
alpha_x_flat = alpha_x_flat[indexes]
u_flat = u_flat[indexes]
v_flat = v_flat[indexes]
PIXELS_TO_ANGLE[v_flat, u_flat] = alpha_x_flat

#plt.imshow(PIXELS_TO_ANGLE)

plt.plot(u_flat, v_flat, 'k.')
#for 

plt.show()
