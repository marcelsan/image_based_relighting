import math

import cv2
import numpy as np
import scipy.ndimage

def ConvertCubeMap2LL(im, outputSize):
	h, w, _ = im.shape

	minSize = max([h,w])//4
	halfMinSize = minSize//2
	out = np.zeros((minSize//2, minSize, 3))

	for x in range(w):
		for y in range(h):
			u = (3.0*(x+1))/w
			v = (4.0*(y+1))/h

			if u>1 and u<2 and v<1: # up
				Vx = (u - 1.5) * 2
				Vy = 1.0
				Vz = (v - 0.5) * -2
			elif u<1 and v>1 and v<2: # left
				Vx = -1.0
				Vy = (v - 1.5) * -2
				Vz = (u - 0.5) * -2
			elif u>1 and u<2 and v>1 and v<2: # forward
				Vx = (u - 1.5) * 2
				Vy = (v - 1.5) * -2
				Vz = -1.0
			elif u>2 and u<3 and v>1 and v<2: # right
				Vx = 1.0
				Vy = (v - 1.5) * -2
				Vz = (u - 2.5) * 2
			elif u>1 and u<2 and v>2 and v<3: # down
				Vx = (u - 1.5) * 2
				Vy = -1.0
				Vz = (v - 2.5) * 2
			elif u>1 and u<2 and v>3 and v<4: # backward
				Vx = (u - 1.5) * 2
				Vy = (v - 3.5) * 2
				Vz = 1.0
			else:
				continue

			normalize = 1.0 / math.sqrt(Vx * Vx + Vy * Vy + Vz * Vz)

			Dx = normalize * Vx
			Dy = normalize * Vy
			Dz = normalize * Vz

			u_ = int((1. + math.atan2(Dx, -Dz)/math.pi)*halfMinSize) - 1
			v_ = int((math.acos(Dy)/math.pi)*halfMinSize) - 1
			out[v_, u_] = im[y, x]

	return cv2.resize(out, (outputSize, outputSize//2), cv2.INTER_CUBIC)

def ConvertLL2Angular(im):
	r, c, _ = im.shape
	maxCoord = max(r, c) / 2
	r = int(maxCoord)
	c = 2*int(maxCoord)

	X0, Y0 = np.meshgrid(np.arange(0, c), np.arange(0, r))

	phi =  np.pi * ((X0 / c) * 2 - 1) - np.pi/2
	theta =  np.pi * (Y0 / r) 

	D = np.zeros((r, c, 3))
	D[:,:,0] = np.cos(phi) * np.sin(theta)
	D[:,:,1] = np.cos(theta)
	D[:,:,2] = np.sin(phi) * np.sin(theta)

	R = (1/np.pi) * np.arccos(D[:,:,2]) / np.sqrt(D[:,:,0]**2 + D[:,:,1]**2)
	X1 = (R * D[:, :, 0])# * c
	Y1 = (R * D[:, :, 1])# * r

	print(X1.min(), X1.max())
	print(Y1.min(), Y1.max())

	imgOut = np.zeros_like(im)
	for i in range(3):
		imgOut[:,:,i] = scipy.ndimage.map_coordinates(im[:,:,i], [X1.ravel(), Y1.ravel()]).reshape(r, c)

	return imgOut