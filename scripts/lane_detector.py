#!/usr/bin/env python

import numpy as np


import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

#import rospy
#from cv_bridge import CvBridge, CvBridgeError
#from sensor_msgs.msg import Image
#from geometry_msgs.msg import Point #geometry_msgs not in CMake file
#from visual_servoing.msg import ConeLocationPixel

import matplotlib.pyplot as plt


class ConeDetector():
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """
    # Color Segmentation 
    LOWER_GRAY = (0, 210, 0) 
    UPPER_GRAY = (255, 270, 255)

    PERCENT_TO_SHOW = 0.20
    STARTING_PERCENT_FROM_TOP = 0.60

    # PERCENT_TO_SHOW = 0.35
    # STARTING_PERCENT_FROM_TOP = 0.35

    def __init__(self):
        # toggle line follower
        self.LineFollower = False

        # Subscribe to ZED camera RGB frames
        # self.cone_pub = rospy.Publisher("/relative_cone_px", ConeLocationPixel, queue_size=10)
        # self.debug_pub = rospy.Publisher("/cone_debug_img", Image, queue_size=10)
        # self.image_sub = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.image_callback)
        # self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

    def debug_mask(self, original_image, blurred_image, masked_image, edge_image, trimmed_edge_image):
        # Plot selected segmentation boundaries
        light_square = np.full((10, 10, 3), self.LOWER_GRAY, dtype=np.uint8) / 255.0
        dark_square = np.full((10, 10, 3), self.UPPER_GRAY, dtype=np.uint8) / 255.0
        plt.subplot(3, 3, 1)
        plt.imshow(hsv_to_rgb(light_square))
        plt.subplot(3, 3, 3)
        plt.imshow(hsv_to_rgb(dark_square))
        
        # Plot Images
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        blurred_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
        masked_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        edge_rgb = cv2.cvtColor(edge_image, cv2.COLOR_BGR2RGB)
        trimmed_rgb = cv2.cvtColor(trimmed_edge_image, cv2.COLOR_BGR2RGB)
        plt.subplot(3, 3, 4)
        plt.imshow(original_rgb)
        plt.subplot(3, 3, 5)
        plt.imshow(blurred_rgb)
        plt.subplot(3, 3, 6)
        plt.imshow(masked_rgb)
        plt.subplot(3, 3, 7)
        plt.imshow(edge_image)
        plt.subplot(3, 3, 8)
        plt.imshow(edge_image)
        plt.show()

    def process_image(self, img, template, debug=False):
        """
        Implement the cone detection using color segmentation algorithm
        Input:
            img: np.3darray; the input image with a cone to be detected. BGR.
            template_file_path; Not required, but can optionally be used to automate setting hue filter values.
        Return:
            bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
                    (x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
        """

        blurred_image = cv2.GaussianBlur(img,(5,5),20)

        # Converts image to hls color space
        hls_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HLS)

        # Filters out colors within our segmentation  boundaries
        mask = cv2.inRange(hls_image, self.LOWER_GRAY, self.UPPER_GRAY)
        masked_image = cv2.bitwise_and(img, img, mask=mask)

        # Converts image to black & white
        _, _, gray_image = cv2.split(masked_image)

        # Runs edge detection
        edge_image = cv2.Canny(gray_image, 100, 200)

        # Cuts out triangle sector in center
        height, width = edge_image.shape
        triangle = np.array([
                        [(100, height), (475, 325), (width, height)]
                        ])
        mask = np.zeros_like(edge_image)
        mask = cv2.fillPoly(mask, triangle, 255)
        trimmed_edge_image = cv2.bitwise_and(edge_image, mask)


        # Displays visual representation of color segmentation process
        if debug==True:
            self.debug_mask(img, blurred_image, masked_image, edge_image, trimmed_edge_image)

        # Return bounding box
        return None

    def image_callback(self, image_msg):
        # Apply your imported color segmentation function (cd_color_segmentation) to the image msg here
        # From your bounding box, take the center pixel on the bottom
        # (We know this pixel corresponds to a point on the ground plane)
        # publish this pixel (u, v) to the /relative_cone_px topic; the homography transformer will
        # convert it to the car frame.

        #################################
        # YOUR CODE HERE
        # detect the cone and publish its
        # pixel location in the image.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        #################################

        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        # Trims image
        if self.LineFollower == True:
            # Add strip of input image to it
            height, width, _ = image.shape

            # Create blank image
            blank_image = np.zeros([height,width,3],dtype=np.uint8)

            start_index = int(height * self.STARTING_PERCENT_FROM_TOP)
            end_index = int(start_index + height * self.PERCENT_TO_SHOW)
            blank_image[start_index: end_index, :] = image[start_index: end_index, :]

            # Output line follower image
            image = blank_image

        # Processes image
        bounding_box = self.process_image(image, ".",False)
        
        bottom_center = ((bounding_box[0][0] + bounding_box[1][0]) / 2, bounding_box[1][1])
        
        # Creates message
        if bounding_box == ((0,0), (0,0)):
            rospy.loginfo("Error: Cone not detected")
        else:
            relative_cone_px = ConeLocationPixel()
            relative_cone_px.u = bottom_center[0]
            relative_cone_px.v = bottom_center[1]

            # Publishes point
            self.cone_pub.publish(relative_cone_px)

        # Debug
        cv2.rectangle(image,bounding_box[0],bounding_box[1],(0,255,0),2)
        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)


if __name__ == '__main__':
    test = ConeDetector()
    image = cv2.imread("../media/start_area.jpg")
    ConeDetector.process_image(test, image, True, True)
    # try:
    #     rospy.init_node('ConeDetector', anonymous=True)
    #     ConeDetector()
    #     rospy.spin()
    # except rospy.ROSInterruptException:
    #     pass
