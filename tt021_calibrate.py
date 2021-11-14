import numpy as np
import cv2
import glob
import os


SELECTED = [24, 27, 30, 31, 37, 40, 43, 45, 46, 48, 49, 52, 53, 59, 64, 68, 74, 75, 81, 85, 88, 89, 95, 100, 106, 108, 109, 110, 114, 115, 116, 123, 133, 140, 142, 149, 150, 151, 154, 192, 164]

DIR = './photos_for_calibration'
OUTPUT_DIR = './photos_for_calibration_process'

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

pattern = (6, 9)
#pattern = (6, 6) # ret = 9.157376474870182

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((pattern[0]*pattern[1],3), np.float32)
objp[:,:2] = np.mgrid[0:pattern[1],0:pattern[0]].T.reshape(-1,2)

#objp = objp[::-1,:]

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

files = sorted(glob.glob(DIR + '/*.png'))
lc = 0
for index in range(len(files)):
    file_ = files[index]
    file_base = os.path.basename(file_)
    file_base_no_ext = os.path.splitext(file_base)[0]
    file_no = int(file_base_no_ext)
    if file_no in SELECTED:
        img = cv2.imread(file_)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, pattern[::-1],None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
        
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
        
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, pattern, corners2,ret)
            #img_disp = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
            #cv2.imshow('img_disp',img_disp)
            #cv2.waitKey()
            
            cv2.imwrite(OUTPUT_DIR + '/' + file_base, img)
        
        lc += 1
        
        #if lc > 10:
        #    break

#cv2.destroyAllWindows()

# https://stackoverflow.com/questions/38815011/error-when-using-two-flags-opencvs-calibratecamera-function
print('start calib...')    

#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None, flags=(cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT)) # ret = 328.17108583023884

#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None, flags=(cv2.CALIB_FIX_PRINCIPAL_POINT)) # ret = 289.8370470896499

camera_matrix = cv2.initCameraMatrix2D(objpoints, imgpoints, gray.shape[::-1])
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_matrix, None, flags=cv2.CALIB_FIX_PRINCIPAL_POINT) # ret = 289.8370470896499
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_matrix, None, flags=cv2.CALIB_RATIONAL_MODEL)
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_matrix, None, flags=cv2.CALIB_THIN_PRISM_MODEL)
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_matrix, None)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_matrix, None, flags=cv2.CALIB_RATIONAL_MODEL)


#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None, flags = cv2.CALIB_FIX_PRINCIPAL_POINT, 	criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)) # ret = 289.8370470896499
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None, flags = cv2.CALIB_FIX_PRINCIPAL_POINT, 	criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 130, 1e-6)) # ret = 289.83205611368663
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None, flags = cv2.CALIB_FIX_PRINCIPAL_POINT, 	criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 130, 1e-8)) # ret = 289.8320561128543
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_matrix, None, flags=cv2.CALIB_RATIONAL_MODEL) 0.76

print('ret =', ret)
print("mtx=np.array(" + str(mtx.tolist()) + ")")
print("dist=np.array(" + str(dist.tolist()) + ")")

# https://stackoverflow.com/questions/29628445/meaning-of-the-retval-return-value-in-cv2-calibratecamera

# to do: trys this:
# https://stackoverflow.com/questions/29371273/input-arguments-of-pythons-cv2-calibratecamera

# 21-08-2020:
#ret = 0.8727215454925135
#mtx=np.array([[617.8249050804442, 0.0, 673.0536941293645], [0.0, 619.3492046143635, 497.9661474464693], [0.0, 0.0, 1.0]])
#dist=np.array([[-0.3123562037471547, 0.1018281655721802, 0.00031297833728767365, 0.0007424882126541622, -0.015160446251882953]]        

# dist.shape = (1, 5)
# k1, k2, p1, p2, k3

    