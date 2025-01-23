import sys
import threading
import numpy as np
import cv2

from . import aim
from . import vex
from .camera import *
from .aim_kin import *
from .evbase import EventRouter
from .events import *
from .actuators import *
from .speech import SpeechListener
from .thesaurus import Thesaurus
from .openai_client import OpenAIClient
from .aruco import *
from .worldmap import *
from .utils import Pose, PoseEstimate
from . import program

class Robot():
    def __init__(self, robot0=None, loop=None, host="192.168.4.1"):
        if robot0 is None:
            robot0 = aim.Robot(host=host)
        robot0.inertial.calibrate()
        robot0.set_pose(0,0,0)
        self.robot0 = robot0
        self.loop = loop
        self.camera = Camera()
        self.kine = AIMKinematics(self)
        self.world_map = WorldMap(self)
        self.aruco_detector = None
        acts = [DriveActuator(self), SoundActuator(self), KickActuator(self), LEDsActuator(self)]
        self.actuators = {act.name : act for act in acts}
        self.erouter = EventRouter(self)
        self.cam_viewer = None
        self.touch = '0x00'
        self.flask_thread = None
        self.openai_client = OpenAIClient(self)
        self.camera_image = None
        self.status = self.robot0._ws_status_thread.current_status['robot']
        robot0._ws_status_thread.callback = self.status_callback
        robot0._ws_img_thread.callback = self.image_callback
        robot0.get_camera_image()  # start the image stream
        robot0.aiv.tag_detection(True)
        self.thesaurus = Thesaurus()
        self.speech_listener = SpeechListener(self, self.thesaurus, debug=False)
        self.loop.call_soon_threadsafe(self.speech_listener.start)

    def status_callback(self):
        self.loop.call_soon_threadsafe(self.status_update)

    def status_update(self):
        self.old_status = self.status
        self.status = self.robot0._ws_status_thread.current_status['robot']
        """
        gyro = sum([abs(float(self.status['gyro_rate'][axis])) for axis in 'xyz'])
        accel = abs(float(self.status['pitch'])) + abs(float(self.status['roll']))
        if accel > 10:
            self.robot0.stop_all_motion()
            self.robot0.play_sound(vex.SoundType.ALARM, 100)
        """
        heading = 360 - self.robot0.get_heading()
        if heading > 180:
            heading = heading - 360
        self.pose = PoseEstimate(self.robot0.get_x(),
                                 -self.robot0.get_y(),
                                 0,
                                 heading / 180 * pi)

        self.battery_percentage = self.status['battery']
        self.update_actuators()
        if self.robot0.is_stopped():
            self.world_map.update()
        t = self.status['touch_flags']
        if self.touch != t:
            #print(f"status_update in {threading.current_thread().native_id}")
            #print(t)
            self.touch = t
            touch_event = TouchEvent(self.status['touch_x'],
                                     self.status['touch_y'],
                                     self.status['touch_flags'])
            self.erouter.post(touch_event)

    def set_pose(self, x, y, z, theta=0):
        self.pose = PoseEstimate(x, y, z, theta)
        # bug preventing this from working
        # self.theta = theta
        x0, y0, theta0 = x, -y, (360 - theta * 180/pi)  # convert to VEX frame
        self.robot0.set_pose(x0, y0, theta0)

    def update_actuators(self):
        for act in self.actuators.values():
            act.status_update()

    def image_callback(self):
        ws = self.robot0._ws_img_thread
        image_bytes = ws.image_list[ws._next_image_index]
        image_array = np.frombuffer(image_bytes, dtype='uint8')
        self.camera_image = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
        if program.running_fsm:
            program.running_fsm.process_image(self.camera_image)

    def turn(self, angle_rads, turn_speed=None):
        if angle_rads > 0:
            turntype = vex.TurnType.LEFT
        else:
            turntype = vex.TurnType.RIGHT
        self.robot0.turn_for(turntype, abs(angle_rads)*180/pi, turn_speed=turn_speed, wait=False)

    def forward(self, distance_mm, drive_speed=None):
        angle_forward = 0
        self.robot0.move_for(distance_mm, angle_forward, drive_speed=drive_speed, wait=False)

    def sideways(self, distance_mm, drive_speed=None):
        angle_leftward = -90
        self.robot0.move_for(distance_mm, angle_leftward, drive_speed=drive_speed, wait=False)

    def is_picked_up(self):
        """
        This function could be smarter about deciding when the robot has been
        put down.  The attitude_threshold should be raised to 7 degrees, but
        the robot should be required to hold its attitude to within 1 degree for
        at least 1 second.  This will handle cases where the robot is put down
        partially on top of something like a pen, so it's a little tilted.
        """
        gyro = self.status['gyro_rate']
        x = float(gyro['x'])
        y = float(gyro['y'])
        gyro_threshold = 15
        pitch = self.robot0.get_pitch()
        roll = self.robot0.get_roll()
        attitude_threshold = 4
        if abs(x) > gyro_threshold or abs(y) > gyro_threshold or \
           abs(pitch) > attitude_threshold or abs(roll) > attitude_threshold:
            #print(f"*** Gyro  x:{x}  y:{y}  pitch:{pitch}  roll:{roll}")
            return True
        else:
            if self.was_picked_up:
                pass # print(f"*** Gyro  x:{x}  y:{y}  pitch:{pitch}  roll:{roll}")
            return False

    def ask_gpt(self, query_text):
        self.openai_client.query(query_text)
