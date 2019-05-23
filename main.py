import math

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyexr

from lib.io import *

parser = argparse.ArgumentParser(description="")
parser.add_argument('--file_name', type=str, required=True)

def ConvertCubeMap2LL(im, outputSize):
	h,w,_ = im.shape

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

	ret = cv2.resize(out, (outputSize, outputSize//2), cv2.INTER_CUBIC)

	return ret

def FromCartesianToSphericalCoordinates(coordinates):
	r,c = coordinates.shape
	ret = np.zeros((r, 2))

	for i, (x,y,z) in enumerate(coordinates):
		r = math.sqrt(x*x + y*y + z*z)
		theta = math.acos(y/r)
		phi = math.atan2(x, z)
		if phi < 0:
			phi += 2*math.pi
		ret[i, :] = (theta, phi)

	return ret

def NormalizeDirections(directions, image_w, image_h):
	xs = []
	ys = []

	for theta, phi in directions:
		x = int((image_w*phi)/(2*math.pi))
		y = int((image_h*theta)/math.pi)

		xs.append(x)
		ys.append(y)

	return xs, ys

def DrawVoronoi(lum, subdiv, directions, light_intensities):
	facets, centers = subdiv.getVoronoiFacetList([])

	weights = np.zeros((len(facets), 3))
	for i in range(0, len(facets)):
		ifacet_arr = []
		for f in facets[i] :
			ifacet_arr.append(f)

		ifacet = np.array(ifacet_arr, np.int)
		ifacets = np.array([ifacet])

		# Obtain mask
		mask = np.zeros_like(lum)
		cv2.fillConvexPoly(mask, ifacet, (255,255,255), cv2.LINE_AA, 0)

		# Compute point integration
		face = cv2.bitwise_and(lum, lum, mask=np.mean(mask, axis=2).astype(np.uint8))
		face_sum = np.array([np.sum(face[:,:,i]) for i in range(3)])
		w = face_sum * math.sin(directions[i][0]) * light_intensities[i]

		weights[i, :] = w

	return weights

if __name__ == "__main__":
	args = parser.parse_args()

	# load environment map HDR image
	im = cv2.imread('data/grace_cross.hdr', -1)[..., ::-1]

	# convert environment map from cube to latitude-longitude mapping 
	lum = ConvertCubeMap2LL(im, 1024)

	# read reflectance field
	reflectance_field, mask = ReadReflectanceField('data/helmet_front_left', 
											'data/helmet_front_left_matte.png')

	# read stage light directions. 
	intensities = ReadLightIntensities('data/light_intensities.txt')

	# read stage light directions.
	directions = ReadLightDirections('data/light_directions.txt')

	# convert light directions from cartesian to spherical coordinates 
	directions = FromCartesianToSphericalCoordinates(directions)

	# normalize light directions to the environment map
	h, w, _ = lum.shape
	xs, ys = NormalizeDirections(directions, w, h)

	# create an instance of Subdiv2D for drawing Voronoi diagram
	subdiv = cv2.Subdiv2D((0, 0, w, h))
	for x, y in zip(xs, ys):
		subdiv.insert((x, y ))

	# draw voronoi diagram
	field_weights = DrawVoronoi(lum, subdiv, directions, intensities)

	# obtain final image
	alpha = np.sum(field_weights, axis=0).max()
	out = np.zeros_like(reflectance_field[0])
	for i in range(len(reflectance_field)):
		out += (reflectance_field[i] * field_weights[i])/alpha

	# change the image background
	h_out, w_out, _ = out.shape
	for i in range(h_out):
		for j in range(w_out):
			if mask[i][j]:
				x = j/float(w_out) - 0.5
				y = -i/float(h_out) + 0.5

				p = np.asarray([x,y,-1]).reshape(1,3)
				theta, phi = FromCartesianToSphericalCoordinates(p)[0]

				u = int((w*phi)/(2*math.pi))
				v = int((h*theta)/math.pi)

				out[i][j] = lum[v][u]

	# save images
	pyexr.write('out/{}.exr'.format(args.file_name), out)