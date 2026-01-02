try: import cv2
except: pass

import math
from math import pi

import numpy as np
from numpy import sqrt, arctan2

from . import camera
from .geometry import *

ARUCO_MARKER_SIZE = 25

class ArucoMarker(object):
    def __init__(self, aruco_parent, marker_id, corners, tvec, rvec):
        self.aruco_parent = aruco_parent
        self.marker_id = marker_id
        self.corners = corners

        # OpenCV Pose information
        self.tvec = tvec
        self.rvec = rvec

        # Marker coordinates in robot's camera reference frame:
        # x points right, y points down, z is depth
        self.camera_coords = (-tvec[0][0], -tvec[1][0], tvec[2][0])

        # Distance along the ground plane; particle filter ignores height so don't include camera y
        self.camera_distance = math.sqrt(tvec[0][0]**2 + tvec[2][0]**2)

        # Conversion to euler angles
        rotationm, jacob = cv2.Rodrigues(rvec)
        self.euler_angles = rotation_matrix_to_euler_angles(rotationm)

    def __str__(self):
        return "<ArucoMarker id=%d tvec=(%d,%d,%d) rvec=(%d,%d,%d) euler=(%d,%d,%d)>" % \
                (self.marker_id, *(self.tvec[:,0].tolist()),
                 *((self.rvec[:,0]*180/pi).tolist()), *self.euler_angles*180/pi)

    def __repr__(self):
        return self.__str__()


class RobotArucoDetector(object):
    def __init__(self, robot, dictionary_name, marker_size=ARUCO_MARKER_SIZE, disabled_ids=[]):
        self.robot = robot
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_name)
        detector_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        self.seen_marker_ids = []
        self.seen_marker_objects = dict()
        self.disabled_ids = disabled_ids  # disable markers with high false detection rates
        self.ids = []
        self.corners = []
        self._last_image_shape = None
        self.marker_size = marker_size #these units will be pose est units!!
        self.object_corners = np.array([
            [-marker_size / 2, marker_size / 2, 0],  # Top left
            [marker_size / 2, marker_size / 2, 0],   # Top right
            [marker_size / 2, -marker_size / 2, 0],  # Bottom right
            [-marker_size / 2, -marker_size / 2, 0]  # Bottom left
            ], dtype=np.float32)

    def process_image(self,gray):
        self.seen_marker_ids = []
        self.seen_marker_objects = dict()
        self._last_image_shape = gray.shape[:2]
        (self.corners, self.ids, _) = self.detector.detectMarkers(gray)
        if self.ids is None: return

        # Estimate poses
        for i in range(len(self.corners)):
            image_corners = self.corners[i]
            id = int(self.ids[i][0])
            try:
                success, rvec, tvec = cv2.solvePnP(self.object_corners,
                                                   image_corners,
                                                   self.robot.camera.camera_matrix,
                                                   self.robot.camera.distortion_array)
            except Exception as e:
                print(f'Aruco detector: solvePnP failed:', e)
                return
            if id in self.disabled_ids: continue
            if rvec[2][0] > math.pi/2 or rvec[2][0] < -math.pi/2:
                # can't see a marker facing away from us, so bogus
                print(f'Marker rejected! id={id}  tvec={tvec.tolist()}  ' +
                      f'rvec={(rvec*180/pi).tolist()}')
                continue
            marker = ArucoMarker(self, id, image_corners, tvec, rvec)
            self.seen_marker_ids.append(id)
            self.seen_marker_objects[id] = marker

    def _scale_corners(self, scale_x, scale_y):
        if scale_x == 1.0 and scale_y == 1.0:
            return self.corners
        scaled = []
        for corner in self.corners:
            scaled_corner = corner.copy()
            scaled_corner[..., 0] = scaled_corner[..., 0] * scale_x
            scaled_corner[..., 1] = scaled_corner[..., 1] * scale_y
            scaled.append(scaled_corner)
        return scaled

    def annotate(self, image, scale_factor=1):
        if image is None or self.ids is None or not self.corners:
            return image
        if not hasattr(cv2, "aruco"):
            return image

        height, width = image.shape[:2]
        scale_x = 1.0
        scale_y = 1.0
        if self._last_image_shape is not None:
            src_h, src_w = self._last_image_shape
            if src_h and src_w and (src_h != height or src_w != width):
                scale_x = width / src_w
                scale_y = height / src_h
        elif scale_factor not in (None, 1, 1.0):
            scale_x = float(scale_factor)
            scale_y = float(scale_factor)

        scaled_corners = self._scale_corners(scale_x, scale_y)

        base = image.copy()
        overlay = np.zeros_like(base)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.aruco.drawDetectedMarkers(overlay_bgr, scaled_corners, self.ids)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        mask = np.any(overlay_rgb != 0, axis=2)
        base[mask] = overlay_rgb[mask]
        displayim = base

        #add poses currently fails since image is already scaled. How to scale camMat?
        #if(self.ids is not None):
        #    for i in range(len(self.ids)):
        #      displayim = cv2.aruco.drawAxis(displayim,self.cameraMatrix,
        #                    self.distortionArray,self.rvecs[i],self.tvecs[i]*scale_factor,self.axisLength*scale_factor)
        return displayim
