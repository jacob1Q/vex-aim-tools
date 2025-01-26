from aim_fsm import *
from cozmo.util import degrees
from aim_fsm.utils import Pose
from aim_fsm.particle import ParticleFilter
from aim_fsm.particle import ArucoDistanceSensorModel, ArucoBearingSensorModel, ArucoCombinedSensorModel
ARUCO_MARKER_SIZE = 50   # millimeters

class lab3(StateMachineProgram):
    def __init__(self):
        landmarks = {
            'ArucoMarker-38.a' : Pose(130.1, 9.4, 20.0, degrees(-167)),
            'ArucoMarker-39.a' : Pose( 137.8, -36.4, 20.0,degrees(-175))
        }

        pf = ParticleFilter(robot,
                            landmarks = landmarks,
                            sensor_model = ArucoDistanceSensorModel(robot)
                            #sensor_model = ArucoBearingSensorModel(robot)
                            #sensor_model = ArucoCombinedSensorModel(robot)
        )

        super().__init__(aruco_marker_size=ARUCO_MARKER_SIZE,
                         particle_filter=pf)
        

'''
C> Added <ArucoMarkerObj 38: (130.4, 5.0, 20.0) @ -167 deg. visible>
Added <ArucoMarkerObj 39: (136.5, -40.8, 20.0) @ -176 deg. visible>

{'ArucoMarker-38.a': <ArucoMarkerObj 38: (130.1, 9.4, 20.0) @ -167 deg. visible>, 'ArucoMarker-39.a': <ArucoMarkerObj 39: (137.8, -36.4, 20.0) @ -175 deg. visible>, 'ArucoMarker-40.a': <ArucoMarkerObj 40: (140.2, -83.1, 20.0) @ -158 deg. visible>}
'''