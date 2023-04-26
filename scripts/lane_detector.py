#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# UNCOMMENT WHEN ON CAR
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
    # Color Segmentation Params
    LOWER_GRAY = (0, 170, 0) 
    UPPER_GRAY = (255, 270, 255)

    # Hough Transform Params
    MIN_SLOPE = 0.25
    LOOKAHEAD_PERCENTAGE = 0.55 # % of screen where 0% is the top of the screen 

    def __init__(self):
        pass
        # UNCOMMENT WHEN ON CAR
        # Subscribe to ZED camera RGB frames
        # self.cone_pub = rospy.Publisher("/relative_cone_px", ConeLocationPixel, queue_size=10)
        # self.debug_pub = rospy.Publisher("/cone_debug_img", Image, queue_size=10)
        # self.image_sub = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.image_callback)
        # self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

    def debug_mask(self, original_image, blurred_image, masked_image, edge_image, triangle_mask, trimmed_edge_image, hough_image, average_hough_image, output_image):
        # Plot selected segmentation boundaries
        light_square = np.full((10, 10, 3), self.LOWER_GRAY, dtype=np.uint8) / 255.0
        dark_square = np.full((10, 10, 3), self.UPPER_GRAY, dtype=np.uint8) / 255.0
        plt.subplot(4, 3, 1)
        plt.imshow(hsv_to_rgb(light_square))
        plt.subplot(4, 3, 3)
        plt.imshow(hsv_to_rgb(dark_square))
        
        # Plot Images
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        blurred_rgb = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)
        masked_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        edge_rgb = cv2.cvtColor(edge_image, cv2.COLOR_BGR2RGB)
        triangle_rgb = cv2.cvtColor(triangle_mask, cv2.COLOR_BGR2RGB)
        trimmed_rgb = cv2.cvtColor(trimmed_edge_image, cv2.COLOR_BGR2RGB)
        hough_rgb = cv2.cvtColor(hough_image, cv2.COLOR_BGR2RGB)
        average_hough_rgb = cv2.cvtColor(average_hough_image, cv2.COLOR_BGR2RGB)
        output_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        plt.subplot(4, 3, 4)
        plt.imshow(original_rgb)
        plt.subplot(4, 3, 5)
        plt.imshow(blurred_rgb)
        plt.subplot(4, 3, 6)
        plt.imshow(masked_rgb)
        plt.subplot(4, 3, 7)
        plt.imshow(edge_image)
        plt.subplot(4, 3, 8)
        plt.imshow(triangle_rgb)
        plt.subplot(4, 3, 9)
        plt.imshow(trimmed_rgb)
        plt.subplot(4, 3, 10)
        plt.imshow(hough_rgb)
        plt.subplot(4, 3, 11)
        plt.imshow(average_hough_rgb)
        plt.subplot(4, 3, 12)
        plt.imshow(output_rgb)
        plt.show()


    # Visualize: Publish average lines to camera graph for real-time visualization (time-cost: small)
    # Debug: Visualize entire image processing process (time-cost: high)
    def process_image(self, img, template, visualize=True, debug=False):
        """
        Implement the cone detection using color segmentation algorithm
        Input:
            img: np.3darray; the input image with a cone to be detected. BGR.
            template_file_path; Not required, but can optionally be used to automate setting hue filter values.
        Return:
            bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
                    (x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
        """

        # Smooths image to reduce noise
        blurred_image = cv2.GaussianBlur(img,(5,5),20)

        # Converts image to hls color space (better for identifying white)
        hls_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HLS)

        # Filters out non-white colors
        mask = cv2.inRange(hls_image, self.LOWER_GRAY, self.UPPER_GRAY)
        masked_image = cv2.bitwise_and(img, img, mask=mask)

        # Converts image to black & white
        _, _, gray_image = cv2.split(masked_image)

        # Runs edge detection
        edge_image = cv2.Canny(gray_image, 100, 200)

        # Cuts out triangle in center
        image_height, image_width = edge_image.shape
        big_triangle = np.array([
                        [(-400, image_height), (np.int32(image_width * 0.5), np.int32(image_height * 0.25)), (image_width+400, image_height)]
                        ])
        # small_triangle = np.array([
        #                 [(135, image_height), (np.int32(image_width * 0.5), np.int32(image_height * 0.40)), (image_width - 135, image_height)]
        #                 ]) # Might not need this part.  Will experiment without it
        triangle_mask = np.zeros_like(edge_image)
        triangle_mask = cv2.fillPoly(triangle_mask, big_triangle, 255)
        # triangle_mask = cv2.fillPoly(triangle_mask, small_triangle, 0) # Might not need this part.  Will experiment without it
        trimmed_edge_image = cv2.bitwise_and(edge_image, triangle_mask)

        # Applies Hough Transform
        lines = cv2.HoughLinesP(trimmed_edge_image, rho=2, theta=np.pi/180, threshold=100, minLineLength=110, maxLineGap=30)
        if lines is None:
            lines = []

        # Visualizes Hough Transform Lines for debugging
        if debug == True:
            hough_image = trimmed_edge_image.copy()
            hough_image = cv2.cvtColor(hough_image, cv2.COLOR_GRAY2BGR)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1)
                if slope < -self.MIN_SLOPE and (x1 + x2) / 2 < image_width / 2:
                    cv2.line(hough_image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                elif slope > self.MIN_SLOPE:
                    cv2.line(hough_image, (x1, y1), (x2, y2), (255, 255, 0), 3)

        # Discards unlikely lines and sorts them
        left_lines = []
        right_lines = []
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            # Only add negative slopes lines with a midpoint in the left of the image
            if parameters[0] < -self.MIN_SLOPE and (x1 + x2) / 2 < image_width / 2:
                left_lines.append(parameters)
            # Only add positive slopes lines with a midpoint in the right of the image
            elif parameters[0] > self.MIN_SLOPE:
                right_lines.append(parameters)


        # If no left line found, make a guess that pulls the car to the left
        if len(left_lines) == 0:
            left_lines.append((-3.0,250))
        # If no right line found, make a guess that pulls the car to the right
        if len(right_lines) == 0:
            right_lines.append((3.0,-1800))

        #finds average lanes
        left_average_line = np.average(left_lines, axis=0)
        right_average_line = np.average(right_lines, axis=0) 

        # Visualize Average Lines for debugging
        left_x0 = 0 
        left_y0 = 0
        right_x0 = 0 
        right_y0 = 0
        intersection_x = 0
        intersection_y = 0

        if visualize or debug:
            intersection_x = int((right_average_line[1] - left_average_line[1]) / (left_average_line[0] - right_average_line[0]))
            intersection_y = int((right_average_line[0]*left_average_line[1] - left_average_line[0]*right_average_line[1]) / (right_average_line[0] - left_average_line[0]))

            left_y0 = image_height
            left_x0 = int((left_y0 - left_average_line[1]) // left_average_line[0])
            right_y0 = image_height
            right_x0 = int((right_y0 - right_average_line[1]) // right_average_line[0])

        if debug == True:
            average_hough_image = trimmed_edge_image.copy()
            average_hough_image = cv2.cvtColor(average_hough_image, cv2.COLOR_GRAY2BGR)
           
            cv2.line(average_hough_image, (left_x0, left_y0), (intersection_x, intersection_y), (255, 0, 255), 3)
            cv2.line(average_hough_image, (right_x0, right_y0), (intersection_x, intersection_y), (255, 255, 0), 3)

        # Finds points to track
        tracking_x = int((right_average_line[1] - left_average_line[1]) / (left_average_line[0] - right_average_line[0])) # Vanishing Point Intersection of the lanes
        tracking_y = int(image_height * self.LOOKAHEAD_PERCENTAGE) 

        # Displays tracking point
        if debug == True:
            output_image = average_hough_image.copy()
            cv2.circle(output_image, (tracking_x, tracking_y), 15, (255, 0, 255), -1)

        # Displays visual representation of color segmentation process
        if debug==True:
            self.debug_mask(img, blurred_image, masked_image, edge_image, triangle_mask, trimmed_edge_image, hough_image, average_hough_image, output_image)

        # Return point to track
        return [(tracking_x, tracking_y), (left_x0, left_y0), (right_x0, right_y0), (intersection_x, intersection_y)]

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

        # Processes image
        tracking_point, left_lane_start, right_lane_start, intersect_point = self.process_image(image, ".",True, False)
        
        # Creates message
        if tracking_point == (0,0):
            rospy.loginfo("Error: Cone not detected")
        else:
            relative_cone_px = ConeLocationPixel()
            relative_cone_px.u = tracking_point[0]
            relative_cone_px.v = tracking_point[1]

            # Publishes point
            self.cone_pub.publish(relative_cone_px)

        # Debug
        cv2.circle(image, tracking_point, 15, (255, 0, 255), -1)
        cv2.line(image, left_lane_start, intersect_point, (255, 0, 255), 3)
        cv2.line(image, right_lane_start, intersect_point, (255, 255, 0), 3)

        debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
        self.debug_pub.publish(debug_msg)


if __name__ == '__main__':
    # UNCOMMENT FOR LOCAL TESTING
    test = ConeDetector()
    image = cv2.imread("../track_images/lane3/image14.png")
    ConeDetector.process_image(test, image, True, True)
    image = cv2.imread("../track_images/lane3/image48.png")
    ConeDetector.process_image(test, image, True, True)
    image = cv2.imread("../track_images/lane3/image3.png")
    ConeDetector.process_image(test, image, True, True)
    image = cv2.imread("../track_images/lane3/image10.png")
    ConeDetector.process_image(test, image, True, True)
    image = cv2.imread("../track_images/lane3/image53.png")
    ConeDetector.process_image(test, image, True, True)
    image = cv2.imread("../track_images/lane1/image7.png")
    ConeDetector.process_image(test, image, True, True)
    image = cv2.imread("../track_images/lane1/image13.png")
    ConeDetector.process_image(test, image, True, True)
    image = cv2.imread("../track_images/lane3/image20.png")
    ConeDetector.process_image(test, image, True, True)
    image = cv2.imread("../track_images/lane3/image17.png")
    ConeDetector.process_image(test, image, True, True)
    image = cv2.imread("../track_images/extra_tests/image1.png")
    ConeDetector.process_image(test, image, True, True)

    # UNCOMMENT WHEN ON CAR
    # try:
    #     rospy.init_node('ConeDetector', anonymous=True)
    #     ConeDetector()
    #     rospy.spin()
    # except rospy.ROSInterruptException:
    #     pass
