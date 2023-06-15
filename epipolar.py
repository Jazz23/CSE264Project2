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


# _______________________________________________________________________________________________________


# Find essential matrix
pts1 = np.float32(pts1)
pts2 = np.float32(pts2)
E, mask = cv2.findEssentialMat(pts1,pts2,mtx)

# Decompose the essential matrix into R_L^R and r^R
R, _, T = cv2.decomposeEssentialMat(E)

pts1 = pts1.reshape(-1,1,2)
pts2 = pts2.reshape(-1,1,2)

# https://stackoverflow.com/questions/66361968/is-cv2-triangulatepoints-just-not-very-accurate
left_projection = mtx @ cv2.hconcat([np.eye(3), np.zeros((3,1))]) # Cam1 is the origin
right_projection = mtx @ cv2.hconcat([R, T])

triangulation = cv2.triangulatePoints(left_projection, right_projection, pts1, pts2)

print(triangulation)

# _______________________________________________________________________________________________________

# Now we have the list of best matches from both the images. Let's find the Fundamental Matrix.
# pts1 = np.int32(pts1)
# pts2 = np.int32(pts2)
# F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

# # We select only inlier points
# pts1 = pts1[mask.ravel()==1]
# pts2 = pts2[mask.ravel()==1]

# # Find epilines corresponding to points in right image (second image) and
# # drawing its lines on left image
# lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
# lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

# # Find epilines corresponding to points in left image (first image) and
# # drawing its lines on right image
# lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
# lines2 = lines2.reshape(-1,3)
# img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

# plt.subplot(121),plt.imshow(img5)
# plt.subplot(122),plt.imshow(img3)
# plt.show()