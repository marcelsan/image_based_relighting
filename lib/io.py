import glob
import os

import cv2
import numpy as np
import pyexr

from lib.geometry import FromCartesianToSphericalCoordinates

def UCSDSortDirectory(e):
	return int(os.path.basename(e).split("_")[0])

def ReadReflectanceField(folder, file_format='png', mask_path=None):
	images = glob.glob(os.path.join(folder, '*'))
	images.sort(key=UCSDSortDirectory)

	ret = []
	for im_file in images:
		if file_format in ['png', 'jpg', 'jpeg', 'hdr']:
			im = cv2.imread(im_file)[..., ::-1] / 255.0
		elif file_format in ['exr']:
			im = pyexr.read_all(im_file)['default']
		else:
			print("No supported file format {}".format(file_format))
			break

		# Undo gamma correction
		no_gamma = np.power(im, 2.2)
		ret.append(no_gamma)

	# Read mask
	mask = None
	if mask_path:
		mask = cv2.imread(mask_path, 0)/255.0

	return ret, mask

def ReadLightDirections(file):
	return np.loadtxt(file)

def ReadLightIntensities(file):
	ret = []

	f = open(file, "r")
	for l in f:
		intensity = [float(c) for c in l.rstrip().split(" ")[1:]]
		ret.append(intensity)

	return np.array(ret)

def ReadLightsInfo(directions_dir, intensities_dir=None):
	directions = ReadLightDirections(directions_dir)
	if intensities_dir:
		intensities = ReadLightIntensities(intensities_dir)
	else:
		intensities = np.ones_like(directions)

	# convert light directions from cartesian to spherical coordinates 
	directions = FromCartesianToSphericalCoordinates(directions)

	return intensities, directions
