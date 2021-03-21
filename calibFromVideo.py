#!/usr/bin/env python3

import numpy as np
import cv2 as cv


fileDir = "./video/720p60_FOV-Ultra.SEC.MP4"
calibFileName = "cam-intrinsics-720p60_FOV-Ultra.yaml"

## Settings
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


## Chess board has 8x8 grid of 35x35 mm squares
cSz = (4,4)         # Chess grid size,  (Dim x, Dim y)
chessXYscale = 35   # In mm. Fixed scale for both directions


objp = np.zeros( (cSz[0]*cSz[1],3), np.float32)
objp[:,:2] = np.mgrid[0:cSz[0],0:cSz[1]].T.reshape(-1,2) * chessXYscale

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


## Open video
cap = cv.VideoCapture(fileDir)
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    #cv.waitKey( round(1000/30) )

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, cSz, None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print("Found chess corners")
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(gray, cSz, corners2, ret)
        cv.imshow('img', gray)
    else:
        print("Did not find chess corners")

    if cv.waitKey(1) == ord('q'):
        break

cap.release()


ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("RMS reprojection error: {}".format(ret))

h,  w = gray.shape[:2]
alpha=0
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), alpha, (w,h))

# undistort
dst = cv.undistort(gray, mtx, dist, None, newcameramtx)
cv.imshow("calibresult", dst)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imshow("calibresult-cropped", dst)


## Save configuration data
s = cv.FileStorage("./calibData/"+calibFileName, cv.FileStorage_WRITE)

s.write('RMSreprojectionError', ret)
s.write('cameraMatrix', mtx)
s.write('distCoeffs', dist)
s.release()


cv.waitKey()

cv.destroyAllWindows()
