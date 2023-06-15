import cv2
import numpy as np

# Load previously saved data
mtx = np.load('mtx.npy')
dist = np.load('dist.npy')
R = np.load('R.npy')
T = np.load('T.npy')

# Calculates matrix H given a depth D
def CalcH(D):
    n = np.array([0,0,-1])
    H = mtx @ (R - (1/D) * T @ n.T) @ np.linalg.inv(mtx)
    return H

