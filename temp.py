# Prints the data saved from calibration
import numpy as np


mtx = np.load('mtx.npy')
dist = np.load('dist.npy')
rvecs = np.load('rvecs.npy')
tvecs = np.load('tvecs.npy')
print(mtx)
print()
print(dist)
print()
print(rvecs)
print()
print(tvecs)
