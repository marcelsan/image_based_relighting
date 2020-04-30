import math

import argparse
import cv2
import numpy as np
import pyexr
import time

from lib.environment_map import ConvertCubeMap2LL
from lib.geometry import FromCartesianToSphericalCoordinates
from lib.io import *
from lib.relighting import ImageRelighting

parser = argparse.ArgumentParser(description="")
parser.add_argument('--file_name', type=str, required=True)
parser.add_argument('--env_dir', type=str, required=True)
parser.add_argument('--light_dirs', type=str, required=True)
parser.add_argument('--ref_field', type=str, required=True)
parser.add_argument('--ref_field_mask', type=str)
parser.add_argument('--light_int', type=str)
parser.add_argument('--cube_to_ll', action='store_true')

if __name__ == "__main__":
	args = parser.parse_args()

	# load environment map HDR image and convert it from cube to 
	# latitude-longitude mapping
	lum = cv2.imread(args.env_dir, -1)[..., ::-1]
	lum = cv2.resize(lum, (256, 128), interpolation=cv2.INTER_AREA)
	h, w, _ = lum.shape
	print(h,w)

	if args.cube_to_ll:
		lum = ConvertCubeMap2LL(lum, 1024)

	# read reflectance field
	reflectance_field, mask = ReadReflectanceField(args.ref_field, 
												mask_path=args.ref_field_mask)
	mask = 1 - mask

	# read stage light directions and intensities
	intensities, directions = ReadLightsInfo(args.light_dirs, args.light_int)

	print("[INFO] Starting image relighting.")
	start = time.time()
	out = ImageRelighting(lum, directions, intensities)\
						.relight(reflectance_field, mask)
	end = time.time()
	print("[INFO] Elapsed time to image relighting: ", end - start)

	# save image
	pyexr.write('out/{}.exr'.format(args.file_name), out)