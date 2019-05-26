import math

import numpy as np

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
	xs, ys = [], []
	
	for theta, phi in directions:
		x = int((image_w*phi)/(2*math.pi))
		y = int((image_h*theta)/math.pi)

		xs.append(x)
		ys.append(y)

	return xs, ys
