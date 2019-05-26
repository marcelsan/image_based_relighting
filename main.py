import math

import argparse
import cv2
import numpy as np
import pyexr

from lib.environment_map import ConvertCubeMap2LL
from lib.geometry import FromCartesianToSphericalCoordinates
from lib.io import *
from lib.relighting import ImageRelighting

parser = argparse.ArgumentParser(description="")
parser.add_argument('--file_name', type=str, required=True)

if __name__ == "__main__":
	args = parser.parse_args()

	# load environment map HDR image and convert it from cube to 
	# latitude-longitude mapping
	im = cv2.imread('data/grace_cross.hdr', -1)[..., ::-1]
	lum = ConvertCubeMap2LL(im, 1024)

	# read reflectance field
	reflectance_field, mask = ReadReflectanceField('data/helmet_front_left', 
											'data/helmet_front_left_matte.png')

	# read stage light directions and intensities
	intensities, directions = ReadLightsInfo('data/light_intensities.txt', 
											'data/light_directions.txt')

	
	out = ImageRelighting(lum, directions, intensities)\
						.relight(reflectance_field, mask)

	# save image
	pyexr.write('out/{}.exr'.format(args.file_name), out)