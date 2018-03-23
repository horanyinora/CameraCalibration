# -*- coding: utf-8 -*-
## problem> our calibration pattern is 7*6 but I cannot use only 6*6 instead
import os

os.chdir("/home/nora")
#directory="./Documents/Hospitalwork/data/calibration/20180314_100131_cricle/calibrationimages/"
directory="./Documents/Hospitalwork/data/calibration/20180314_110050_circle/calibrationimages/"

import numpy as np
import cv2
import glob
import yaml
import matplotlib.pyplot as plt

def inverte(imagem):
    imagem = (255-imagem)
    return imagem

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

########################################Blob Detector##############################################

# Setup SimpleBlobDetector parameters.
blobParams = cv2.SimpleBlobDetector_Params()

# Change thresholds
blobParams.minThreshold = 8
blobParams.maxThreshold = 255

# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 300     # minArea may be adjusted to suit for your experiment
blobParams.maxArea = 2500   # maxArea may be adjusted to suit for your experiment

# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.1

# Filter by Convexity
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.87

# Filter by Inertia
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.01

# Create a detector with the parameters
blobDetector = cv2.SimpleBlobDetector_create(blobParams)

###################################################################################################

# Original blob coordinates, supposing all blobs are of z-coordinates 0
# And, the distance between every two neighbour blob circle centers is 72 centimetres
# In fact, any number can be used to replace 72.
# Namely, the real size of the circle is pointless while calculating camera calibration parameters.

objp = np.zeros((42, 3), np.float32)

objp[0]  = (0  , 0  , 0)
objp[1]  = (0  , 0.4 , 0)
objp[2]  = (0  , 0.8, 0)
objp[3]  = (0  , 0.12, 0)
objp[4]  = (0  , 0.16, 0)
objp[5]  = (0  , 0.20, 0)
objp[6]  = (0  , 0.24, 0)
objp[7]  = (0.4 , 0  , 0)
objp[8]  = (0.4 , 0.4 , 0)
objp[9] = (0.4 , 0.8, 0)
objp[10] = (0.4 , 0.12, 0)
objp[11] = (0.4, 0.16,  0)
objp[12] = (0.4, 0.20, 0)
objp[13] = (0.4, 0.24, 0)
objp[14] = (0.8, 0  , 0)
objp[15] = (0.8, 0.4 , 0)
objp[16] = (0.8, 0.8, 0)
objp[17] = (0.8, 0.12, 0)
objp[18] = (0.8, 0.16 , 0)
objp[19] = (0.8, 0.20, 0)
objp[20] = (0.8, 0.24, 0)
objp[21] = (0.12, 0, 0)
objp[22] = (0.12, 0.4  , 0)
objp[23] = (0.12, 0.8 , 0)
objp[24] = (0.12, 0.12, 0)
objp[25] = (0.12, 0.16, 0)
objp[26] = (0.12, 0.20 , 0)
objp[27] = (0.12, 0.24, 0)
objp[28] = (0.16, 0  , 0)
objp[29] = (0.16, 0.4 , 0)
objp[30] = (0.16, 0.8, 0)
objp[31] = (0.16, 0.12, 0)
objp[32] = (0.16, 0.16 , 0)
objp[33] = (0.16, 0.20, 0)
objp[34] = (0.16, 0.24, 0)
objp[35] = (0.20, 0, 0)
objp[36] = (0.20, 0.4  , 0)
objp[37] = (0.20, 0.8 , 0)
objp[38] = (0.20, 0.12, 0)
objp[39] = (0.20, 0.16  , 0)
objp[40] = (0.20, 0.20, 0)
objp[41] = (0.20, 0.24, 0)


###################################################################################################

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(directory + "*.jpg")
#images = glob.glob("./Documents/Hospitalwork/data/calibration/20180314_101337_chess/calib_convert/*.jpg")
i=0
for fname in images: #images
     print ("processing %s..." % fname)
     img = cv2.imread(fname)
     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
     
     if np.sum(gray) > 70000000:
         gray=gray
#         plt.imshow(gray, cmap='gray');plt.show()
     else:
         gray = inverte(gray)
#         plt.imshow(gray, cmap='gray');plt.show()

     ret, corners = cv2.findCirclesGrid(gray, (6,6), None, flags = cv2.CALIB_CB_SYMMETRIC_GRID)   # Find the circle grid
     # If found, add object points, image points (after refining them)
     if ret == True:
         i=i+1
         corners2 = cv2.cornerSubPix(image=gray, corners=corners, winSize=(11,11), zeroZone=(-1,-1),
                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
         imgpoints.append(corners2)
         objpoints.append(objp[0:corners2.shape[0]])
         
         # Draw and display the corners
         cv2.drawChessboardCorners(img, (6,6), corners2,ret)
         print ("Success! %i..." % i)
         plt.imshow(img);plt.show()
         
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

mean_error = 0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print "total error: ", mean_error/len(objpoints)

# Write result into file
import time
timestr = time.strftime("%Y%m%d_%H%M%S")

data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
with open(directory + "calibration_"+timestr+".yaml", "w") as f:
    yaml.dump(data, f)
#    
#    
with open(directory + "calibration_"+timestr+".yaml") as f:
    loadeddict = yaml.load(f)
    mtxloaded = loadeddict.get('camera_matrix')
    distloaded = loadeddict.get('dist_coeff')




