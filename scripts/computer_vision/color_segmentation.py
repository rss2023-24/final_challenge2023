import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import pdb

# Sources consulted:
# https://realpython.com/python-opencv-color-spaces/
# https://medium.com/srm-mic/color-segmentation-using-opencv-93efa7ac93e2 
# https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/
# https://stackoverflow.com/questions/44588279/find-and-draw-the-largest-contour-in-opencv-on-a-specific-color-python

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

## Color segmentation boundaries

# Cone Values
#DARK_ORANGE = (1, 190, 90) # Make final number smaller to detect darker oranges
#LIGHT_ORANGE  = (26, 255, 255) # Make first number larger to detect yellows

DARK_ORANGE = (1, 190, 0) # Make final number smaller to detect darker oranges
LIGHT_ORANGE  = (100, 255, 255) # Make first number larger to detect yellows

# DARK_ORANGE = (6, 30, 50) # Make final number smaller to detect darker oranges
# LIGHT_ORANGE  = (15, 60, 100) # Make first number larger to detect yellows

def image_print(img):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow("image", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Creates visual representation of color segmentation process
def debug_mask(original_image, filtered_image, masked_image):
	# Plot selected segmentation boundaries
	light_square = np.full((10, 10, 3), LIGHT_ORANGE, dtype=np.uint8) / 255.0
	dark_square = np.full((10, 10, 3), DARK_ORANGE, dtype=np.uint8) / 255.0
	plt.subplot(2, 3, 1)
	plt.imshow(hsv_to_rgb(light_square))
	plt.subplot(2, 3, 3)
	plt.imshow(hsv_to_rgb(dark_square))
	
	# Plot Images
	original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
	filtered_rgb = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
	masked_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
	plt.subplot(2, 3, 4)
	plt.imshow(original_rgb)
	plt.subplot(2, 3, 5)
	plt.imshow(filtered_rgb)
	plt.subplot(2, 3, 6)
	plt.imshow(masked_rgb)
	plt.show()

def cd_color_segmentation(img, template, debug=False):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; Not required, but can optionally be used to automate setting hue filter values.
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	kernel = np.ones((5, 5), np.uint8)

	# Performs filtering to remove image noise
	# TODO This section can be tuned to produce better scores

	# filter1_image = cv2.blur(img,(5,5))
	# filter2_image = cv2.medianBlur(filter1_image,5)
	# filter3_image = cv2.GaussianBlur(filter2_image,(5,5),0)
	# filter4_image = cv2.bilateralFilter(filter3_image,9,75,75)
	# erosion_image = cv2.erode(filter4_image, kernel, iterations=2)
	# dilation_image = cv2.dilate(erosion_image, kernel, iterations=2)

	filtered_image = cv2.GaussianBlur(img,(5,5),0)

	# Converts image to hsv color space
	hsv_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2HSV)

	# Filters out colors within our segmentation  boundaries
	mask = cv2.inRange(hsv_image, DARK_ORANGE, LIGHT_ORANGE)
	masked_image = cv2.bitwise_and(img, img, mask=mask)

	# Finds countours	
	ret,thresh = cv2.threshold(mask, 40, 255, 0)
	_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	# Draw biggest bounding box
	bounding_box = ((0,0),(0,0))
	if len(contours) != 0:
		# Draws biggest contour
		# cv2.drawContours(masked_image, contours, -1, 255, 3)

		# new_contours = []
		
		# # Filters out unreasonable contours
		# for contour in contours:
		# 	(x,y,w,h) = cv2.boundingRect(contour)
		# 	if not (w > h + 10):
		# 		new_contours.append(contour)

		# find the biggest countour by area
		c = max(contours, key = cv2.contourArea)
		x,y,w,h = cv2.boundingRect(c)
		bounding_box = ((x,y),(x+w,y+h))

		# draw bounding box around biggest contour
		cv2.rectangle(masked_image,bounding_box[0],bounding_box[1],(0,255,0),2)

	# Displays visual representation of color segmentation process
	if debug==True:
		debug_mask(img, filtered_image, masked_image)
		print(bounding_box)

	# Return bounding box
	return bounding_box


if __name__ == "__main__":
    image = cv2.imread("./test_images_cone/test14.jpg")
    cd_color_segmentation(image, ".")
