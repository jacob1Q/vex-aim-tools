import numpy as np

class Camera():
    def __init__(self):
        focal_length = (400, 400)
        resolution = (640, 480)
        center = (320, 240) # should be adjusted for each robot's camera chip

        self.focal_length = focal_length
        self.resolution = resolution
        self.focal_length = focal_length
        self.center = center
        self.fov_x = 70 # degrees (estimated)
        self.fov_y = 50 # degrees (estimated)

        # These are used by cv2.SolvePnP:
        self.camera_matrix = \
            np.array([[focal_length[0],   0,                center[0]],
                      [0,               -focal_length[1],   center[1]],
                      [0,                 0,                 1       ]]).astype(float)
        self.distortion_array = np.array([0,0,0,0,0]).astype(float)
