import math

import cv2
import numpy as np

from lib.geometry import FromCartesianToSphericalCoordinates
from lib.geometry import NormalizeDirections

class ImageRelighting(object):
	
	def __init__(self, lum, directions, intensities):
		self.lum = lum
		self.directions = directions
		self.light_intensities = intensities

		# normalize light directions to the environment map
		h, w, _ = lum.shape
		xs, ys = NormalizeDirections(directions, w, h)

		# create an instance of Subdiv2D for drawing Voronoi diagram
		self.subdiv = cv2.Subdiv2D((0, 0, w, h))
		for x, y in zip(xs, ys):
			self.subdiv.insert((x, y))

	def relight(self, reflectance_field, mask):
		field_weights = self._get_field_weight()
		
		alpha = np.sum(field_weights, axis=0).max()
		out = np.zeros_like(reflectance_field[0])
		for i in range(len(reflectance_field)):
			out += (reflectance_field[i] * field_weights[i])/alpha

		out = self._get_background(out, mask)

		return out

	def _get_background(self, out, mask):
		h_out, w_out, _ = out.shape
		h, w, _ = self.lum.shape

		for i in range(h_out):
			for j in range(w_out):
				if mask[i][j]:
					x = j/float(w_out) - 0.5
					y = -i/float(h_out) + 0.5

					p = np.asarray([x,y,-1]).reshape(1,3)
					theta, phi = FromCartesianToSphericalCoordinates(p)[0]

					u = int((w*phi)/(2*math.pi))
					v = int((h*theta)/math.pi)
					out[i][j] = self.lum[v][u]

		return out

	def _get_field_weight(self):
		facets, _ = self.subdiv.getVoronoiFacetList([])
		weights = np.zeros((len(facets), 3))

		for i in range(0, len(facets)):
			ifacet_arr = []
			for f in facets[i] :
				ifacet_arr.append(f)

			ifacet = np.array(ifacet_arr, np.int)
			ifacets = np.array([ifacet])

			# Compute point integration
			mask = np.zeros_like(self.lum)
			cv2.fillConvexPoly(mask, ifacet, (255,255,255), cv2.LINE_AA, 0)
			face = cv2.bitwise_and(self.lum, self.lum, mask=np.mean(mask, axis=2).astype(np.uint8))
			face_sum = np.array([np.sum(face[:,:,i]) for i in range(3)])
			w = face_sum * math.sin(self.directions[i][0]) * self.light_intensities[i]
			weights[i, :] = w

		return weights