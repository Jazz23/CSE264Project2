import cv2
import numpy as np

# Load previously saved data
mtx = np.load('mtx.npy')
dist = np.load('dist.npy')
pts1 = np.load('pts1.npy')
pts2 = np.load('pts2.npy')

# Find essential matrix
pts1 = np.float32(pts1)
pts2 = np.float32(pts2)

# Draw the pts1 onto img1
img1 = cv2.imread('pair/left.jpg')

for i in range(len(pts1)):
    cv2.circle(img1, (int(pts1[i][0]), int(pts1[i][1])), 5, (255, 0, 0), -1)


# Find essential matrix
E, mask = cv2.findEssentialMat(pts1,pts2,mtx)

# Decompose the essential matrix into R_L^R and r^R
R, _, T = cv2.decomposeEssentialMat(E)

# Save R and T to files
np.save('R.npy', R)
np.save('T.npy', T)

# Format points for triangulation
pts1 = pts1.reshape(-1,1,2)
pts2 = pts2.reshape(-1,1,2)

# https://stackoverflow.com/questions/66361968/is-cv2-triangulatepoints-just-not-very-accurate
left_projection = mtx @ cv2.hconcat([np.eye(3), np.zeros((3,1))]) # Cam1 is the origin
right_projection = mtx @ cv2.hconcat([R, T])

points4d = cv2.triangulatePoints(left_projection, right_projection, pts1, pts2)
points3d = (points4d[:3, :]/points4d[3, :]).T

# Multiply each point in points3d by mtx to get the pixel coords,
# then draw them onto img1
for point in points3d:
    point = point.reshape(-1,1,3)
    point = cv2.projectPoints(point, np.zeros((3,1)), np.zeros((3,1)), mtx, dist)[0]
    point = point.reshape(-1,2)
    cv2.circle(img1, (int(point[0][0]), int(point[0][1])), 5, (0, 255, 0), -1)

cv2.imshow('img1', img1)
cv2.waitKey(0)