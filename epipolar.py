import cv2
import numpy as np
from matplotlib import pyplot as plt


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

img1 = cv2.imread('pair/left.jpg',0)  #queryimage # left image
img2 = cv2.imread('pair/right.jpg',0) #trainimage # right image


# Get the camera matrix and distortion coefficients from calibration
mtx = np.load('mtx.npy')
dist = np.load('dist.npy')

# Undistort points using the camera matrix and distortion coefficients
h, w = img1.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
# undistort
img1 = cv2.undistort(img1, mtx, dist, None, newcameramtx)
img2 = cv2.undistort(img2, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
img1 = img1[y:y+h, x:x+w]
img2 = img2[y:y+h, x:x+w]

# find the keypoints and descriptors with SIFT

sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
pts1 = []
pts2 = []

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

# Save the points to files
np.save('pts1.npy', pts1)
np.save('pts2.npy', pts2)