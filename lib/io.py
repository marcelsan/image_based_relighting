import glob

import cv2
import numpy as np
import pyexr

def ReadReflectanceField(folder, mask_path):
	images = glob.glob(folder + '/*')
	ret = [pyexr.read_all(file)['default'] for file in images]

	# Read mask
	mask = cv2.imread(mask_path, 0)/255.0

	return ret, mask

def ReadLightDirections(file):
	directions = []

	f = open(file, "r")
	for l in f:
		pos = [-float(c) for c in l.rstrip().split(" ")[1:]]
		directions.append(pos)

	return np.array(directions)

def ReadLightIntensities(file):
	ret = []

	f = open(file, "r")
	for l in f:
		intensity = [float(c) for c in l.rstrip().split(" ")[1:]]
		ret.append(intensity)

	return np.array(ret)