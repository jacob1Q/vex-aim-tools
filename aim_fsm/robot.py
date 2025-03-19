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
from .rrt import RRT
from .path_planner import PathPlanner
from .utils import Pose, PoseEstimate
from . import program

class Robot():
    def __init__(self, robot0=None, loop=None, host="192.168.4.1"):
        if robot0 is None:
            robot0 = aim.Robot(host=host)
        robot0.inertial.calibrate()
        robot0.set_pose(0,0,0)
        self.pose = Pose(0,0,0,0)
        self.robot0 = robot0
        self.loop = loop
        self.camera = Camera()
        self.kine = AIMKinematics(self)
        self.world_map = WorldMap(self)
        self.worldmap_viewer = None
        self.rrt = RRT(self)
        self.path_planner= PathPlanner()
        self.aruco_detector = None
        acts = [DriveActuator(self), SoundActuator(self), KickActuator(self), LEDsActuator(self)]
        self.actuators = {act.name : act for act in acts}
        self.erouter = EventRouter(self)
        self.particle_filter = None
        self.particle_viewer = None
        self.path_viewer = None
        self.cam_viewer = None
        self.touch = '0x00'
        self.flask_thread = None
        self.openai_client = OpenAIClient(self)
        self.camera_image = None
        self.frame_count = 0    # camera images received so far
        self.moving_frame = 0   # last camera image when robot was moving
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
        theta = heading / 180 * pi
        self.pose = PoseEstimate(self.robot0.get_y(),
                                 -self.robot0.get_x(),
                                 0,
                                 theta)

        self.battery_percentage = self.status['battery']
        self.update_actuators()
        if not self.robot0.is_stopped():
            self.moving_frame = self.frame_count
        else:
            if self.frame_count > self.moving_frame + 1:
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

    def set_pose(self, x, y, z, theta, reset_particles=True):
        self.pose = PoseEstimate(x, y, z, theta)
        x0, y0, heading0 = -y, x, (360 - theta * 180/pi)  # convert to VEX frame
        self.robot0.set_pose(x0, y0, heading0)
        if self.particle_filter and reset_particles:
            self.particle_filter.set_pose(x,y,theta)

    def update_actuators(self):
        for act in self.actuators.values():
            act.status_update()

    def clear_actuators(self):
        for act in self.actuators.values():
            act.clear()

    def image_callback(self):
        ws = self.robot0._ws_img_thread
        image_bytes = ws.image_list[ws._next_image_index]
        image_array = np.frombuffer(image_bytes, dtype='uint8')
        self.camera_image = cv2.cvtColor(cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        self.frame_count += 1
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

    def move(self, distance_mm, angle_rads, drive_speed=None, turn_speed=None):
        self.robot0.move_for(distance_mm, angle_rads*180/pi,
                             drive_speed=drive_speed, turn_speed=turn_speed, wait=False)

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

    def is_moving(self):
        return not self.robot0.is_stopped()

    def ask_gpt(self, query_text):
        self.openai_client.query(query_text)

    def send_gpt_camera(self, instruction=None):
        self.openai_client.send_camera_image(instruction=instruction)

    def ask_gpt_camera(self, query_text):
        self.openai_client.camera_query(query_text)

    def gpt_oneshot(self, query_text, image=None):
        self.openai_client.oneshot_query(query_text, image)

    def show_pose(self):
        def neaten(x):
            return round(x*10)/10
        print(f'Odometry:  {neaten(self.pose.x)}, ' +
              f'{neaten(self.pose.y)} ' +
              f'heading {neaten(self.pose.theta*180/pi)} deg.', end='')
        print(f'   [ Roll: {neaten(self.robot0.get_roll())}  ' +
              f'Pitch: {neaten(self.robot0.get_pitch())}  ' +
              f'Yaw: {neaten(self.robot0.get_yaw())} ]')
        pf_pose = self.particle_filter.update_pose_estimate()
        print(f'Particles: {neaten(pf_pose.x)}, ' +
              f'{neaten(pf_pose.y)} ' +
              f'heading {neaten(pf_pose.theta*180/pi)} deg.  ' +
              f'[{self.particle_filter.state}]')
        print()

    def print_raw_odometry(self):
        print(f'x={self.robot0.get_x()} y={self.robot0.get_y()} hdg={self.robot0.get_heading()}')
