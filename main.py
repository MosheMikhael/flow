# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:08:10 2018

@author: moshe.f
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

import draw
import point_at_infinity as pai
import service_functions as sf

X, Y = 800, 600
dx, dy = 10, 10
Nx, Ny = int(X / dx), int(Y / dy)
N = Nx * Ny
grid = []
    
for i in range(Nx):
    for j in range(Ny):
        grid.append([int(i * dx + dx / 2), int(j * dy + dy / 2)])
grid = np.array(grid)
grid_m = sf.modification_points(grid)

cam = cv2.VideoCapture(0)
is_read, img1 = cam.read()
is_read, img2 = cam.read()
img1 = cv2.resize(img1, (X, Y)) 
img2 = cv2.resize(img2, (X, Y)) 
while is_read:
    out = img2
    pts1, pts2 = pai.find_opt_flow_lk_with_points(img1, img2, grid_m, 15, 15)
    norms = [np.linalg.norm(pts1[i] - pts2[i]) for i in range(len(pts1))]
    n_max = max(norms)
    n_min = min(norms)
#    gray = np.zeros((Y, X, 3), int)
    
    for i in range(N):
        out = draw.draw_point(out, grid[i], radius=2)
    for i in range(len(pts1)):
        out = draw.draw_arrow(out, pts1[i], pts2[i])
        gray[pts1[i][1]][pts1[i][2]] != (norms[i] - n_min, norms[i] - n_min, norms[i] - n_min)
    cv2.imshow('cam', out)        
#    cv2.imshow('gray', gray)        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    img1 = img2
    is_read, img2 = cam.read()
    
    img2 = cv2.resize(img2, (X, Y)) 
# When everything is done, release the capture
cam.release()
cv2.destroyAllWindows()
