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
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_matrix, None, flags=cv2.CALIB_RATIONAL_MODEL)


#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None, flags = cv2.CALIB_FIX_PRINCIPAL_POINT, 	criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)) # ret = 289.8370470896499
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None, flags = cv2.CALIB_FIX_PRINCIPAL_POINT, 	criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 130, 1e-6)) # ret = 289.83205611368663
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None, flags = cv2.CALIB_FIX_PRINCIPAL_POINT, 	criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 130, 1e-8)) # ret = 289.8320561128543
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_matrix, None, flags=cv2.CALIB_RATIONAL_MODEL) 0.76

# 2021-02-14:
#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_matrix, None, flags=cv2.CALIB_RATIONAL_MODEL, criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 130, 1e-8))
#ret = 0.7685083054688858
#mtx=np.array([[613.514017411662, 0.0, 683.6086788170371], [0.0, 614.9188816408458, 499.2844922051985], [0.0, 0.0, 1.0]])
#dist=np.array([[-1.0608154255688673, 0.19568544091175313, 0.000537101024529654, 0.0001049916307467493, 0.053879036934391195, -0.7213533273337543, -0.2141713431279143, 0.17998481314637724, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_matrix, None, flags=cv2.CALIB_THIN_PRISM_MODEL)
#ret = 0.9434123461976375
#mtx=np.array([[614.753395445264, 0.0, 656.4801230296806], [0.0, 616.8134418714153, 494.4874248567994], [0.0, 0.0, 1.0]])
#dist=np.array([[-0.31846825600710965, 0.11149055734041727, -0.0008971840280663076, -0.003337373148996942, -0.01894951025877752, 0.0, 0.0, 0.0, 0.013568738978704817, -0.0002203914809455231, 0.002009319389872685, 0.0005239117038633903]])

#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_matrix, None, flags=cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL)
#ret = 0.7585691178370598
#mtx=np.array([[616.216941335761, 0.0, 676.4908636515407], [0.0, 617.1366312504531, 530.5681049561929], [0.0, 0.0, 1.0]])
#dist=np.array([[-1.1269606631511337, 0.22228484542127555, 0.0049154225144352245, -0.0002912859369946006, 0.06317823645404994, -0.7829415613064403, -0.21837220245608102, 0.206830100550455, 0.003653985959286356, -0.0006982421696678767, -0.018892678275357754, 0.0025828333327129562]])

#ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_matrix, None, flags=cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL + cv2.CALIB_TILTED_MODEL)
#ret = 0.7672996719008202
#mtx=np.array([[616.0437365784716, 0.0, 640.8639070158933], [0.0, 616.5617628889557, 557.3987187948529], [0.0, 0.0, 1.0]])
#dist=np.array([[-0.4007673145500792, 1.0469972022132086, -0.00424357048257913, -0.00018714618896861784, 0.1671949108193266, -0.03905960832175742, 0.8271528968660912, 0.5636727796551333, 0.021095478865837645, -0.003809881826330748, -0.022163088080381692, 0.005717953758756432, 0.03768366504582744, 0.01932154278639963]])

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], camera_matrix, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)
#ret = 0.873084271040077
#mtx=np.array([[618.7162038517924, 0.0, 671.9856848350533], [0.0, 620.20895334707, 497.72866882850883], [0.0, 0.0, 1.0]])
#dist=np.array([[-0.3132964900492584, 0.10279928422337736, 0.0003079750471391313, 0.0007692445996596191, -0.015460211904764277]])

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

# 2021-02-05, cv2.CALIB_RATIONAL_MODEL :
#ret = 0.7685575129516239
#mtx=np.array([[613.6206796762666, 0.0, 683.4888941175157], [0.0, 615.0136787931489, 499.2466309019339], [0.0, 0.0, 1.0]])
#dist=np.array([[-1.047488351559334, 0.1895766552719301, 0.0005408151134692955, 0.00010288485616716016, 0.053297712776547455, -0.707423458071405, -0.2169199947848843, 0.17741744871291024, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    