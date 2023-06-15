path = ''

import cv2
import numpy as np

# Load previously saved data for convient execution
mtx = np.load(path + 'mtx.npy')
dist = np.load(path + 'dist.npy')
R = np.load(path + 'R.npy')
T = np.load(path + 'T.npy')
points3d = np.load(path + 'points3d.npy')
img1 = cv2.imread(path + 'pair/left.jpg',0)
img2 = cv2.imread(path + 'pair/right.jpg',0)

# Calculates inverse of the matrix H given a depth D
def CalcHInverse(D):
    # n is a column vector
    n = np.array([[0], [0], [1]])
    H = mtx @ (R - (1/D) * (T @ n.T)) @ np.linalg.inv(mtx)
    return np.linalg.inv(H)

def GetWarped(depth):
    Hinv = CalcHInverse(depth)

    # Warp img2 using homography Hinv
    img2_warped = cv2.warpPerspective(img2, Hinv, (img2.shape[1], img2.shape[0]))
    return img2_warped

def Overlay(warped):
    # Overlay img2 onto img1, giving img2 a transparent effect but keep img1 at full transparency
    img1_overlayed = cv2.addWeighted(img1, 1, warped, 0.5, 0)

    # Scale the image to fit a little better
    scale_percent = 25 # percent of original size
    width = int(img1_overlayed.shape[1] * scale_percent / 100)
    height = int(img1_overlayed.shape[0] * scale_percent / 100)
    dim = (width, height)
    img1_overlayed = cv2.resize(img1_overlayed, dim, interpolation = cv2.INTER_AREA)
    return img1_overlayed


# Find the max value of the z component of the points in points3d, an Nx3 array
min = 100
max = 0
for point in points3d:
    if point[2] > max and point[2]:
        max = point[2]
    if point[2] < min and point[2] > 0:
        min = point[2]

print("Min depth: ", min)
print("Max depth: ", max)

# Gather warped images using min/max depth
warped = []
delta = (max - min) / 20
for i in range(0, 20):
    warped_img = GetWarped(min + i * delta)
    warped.append(warped_img) # NOT WORKING. The alignment is way off, no clue why
    overlayed = Overlay(warped_img)
    # cv2.imshow('bruh', overlayed)
    # cv2.waitKey(0)
    cv2.imwrite(path + f'warped/warped{i}.jpg', overlayed)

# Run the block filter on the left image
filtered_image = cv2.boxFilter(img1, -1, (15, 15))

# Apply the block filter to each warped image
for i in range(len(warped)):
    warped[i] = cv2.boxFilter(warped[i], -1, (15, 15))

# For each pixel in filtered_image, find the absolute differnece in brightness
# with the corrosponding pixel in each warped image. Record index (depth) of
# the warped image with the lowest difference
depths = np.zeros((filtered_image.shape[0], filtered_image.shape[1]))
for i in range(filtered_image.shape[0]):
    for j in range(filtered_image.shape[1]):
        min_diff = 255
        min_depth_index = min
        for k in range(len(warped)):
            if i < warped[k].shape[0] and j < warped[k].shape[1]:
                diff = abs(int(filtered_image[i][j]) - int(warped[k][i][j]))
                if diff < min_diff:
                    min_diff = diff
                    min_depth_index = k
        depth =  min + min_depth_index * delta # Multiply index by delta to get depth
        b = (depth - min) / (max - min) * 255 # Normalize depth to 0-255 for brightness
        # Assign pixel i j in depths to be (b, b, b)
        depths[i][j] = b

# Display the depth map
cv2.imshow(depths)
cv2.waitKey(0)