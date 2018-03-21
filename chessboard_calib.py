import os
os.chdir("/home/nora")
directory="./Documents/Hospitalwork/data/calibration/20180314_101337_chess/calibrationimages/"
#directory="./Documents/Hospitalwork/data/calibration/20180314_105951_chess/calibrationimages/"

import numpy as np
import cv2
import glob
import yaml
import matplotlib.pyplot as plt
import time

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

objp = np.zeros((4*4,3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:4].T.reshape(-1,2)

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
     gray = inverte(gray)
     plt.imshow(gray, cmap='gray');plt.show()
     
     # Find the chess board corners
     ret, corners = cv2.findChessboardCorners(gray, (4,4), None) #cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    
     # If found, add object points, image points (after refining them)
     if ret == True:
         i=i+1
         corners2 = cv2.cornerSubPix(image=gray, corners=corners, winSize=(11,11), zeroZone=(-1,-1),
                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
         imgpoints.append(corners2)
         objpoints.append(objp[0:corners2.shape[0]])
         
         # Draw and display the corners
         cv2.drawChessboardCorners(img, (4,4), corners2,ret)
         plt.imshow(img);plt.show()
         
cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

# Write result into file
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





