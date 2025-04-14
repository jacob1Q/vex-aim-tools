from math import pi
import re
from importlib import __import__, reload
try:
    from termcolor import cprint
except:
    def cprint(string,color=None):
        print(string)

import cv2

from . import vex
from . import evbase

from .evbase import EventRouter
from .base import StateNode
from .cam_viewer import CamViewer
from .worldmap_viewer import WorldMapViewer
from .aruco import *
from .worldmap import WorldMap
from .particle import *
from .utils import Pose
from .particle_viewer import ParticleViewer
from .rrt import RRT
from .path_viewer import PathViewer
from . import opengl
#from . import custom_objs
#from .perched import *
#from .sharedmap import *

running_fsm = None

class StateMachineProgram(StateNode):
    def __init__(self,
                 launch_cam_viewer = True,
                 launch_worldmap_viewer = True,
                 force_annotation = False,   # set to True for annotation even without cam_viewer
                 annotate_sdk = True,        # include annotations for SDK's object detections
                 annotated_scale_factor = 1, # set to 1 to avoid cost of resizing images
                 viewer_crosshairs = False,  # set to True to draw viewer crosshairs
                 speech = True,

                 particle_filter = None,
                 num_particles = 500,
                 sensor_model = "default",
                 landmark_test = SLAMSensorModel.is_wall_landmark, # SLAMSensorModel.is_solo_aruco_landmark, #
                 landmarks = None,
                 launch_particle_viewer = False,
                 particle_viewer_scale = 1.0,
                 launch_path_viewer = False,
                 aruco = True,
                 dictionary_name = cv2.aruco.DICT_4X4_100,
                 aruco_disabled_ids = (17, 37),
                 aruco_marker_size = ARUCO_MARKER_SIZE,

                 perched_cameras = False,

                 rrt = None,
                 ):
        super().__init__()
        self.name = self.__class__.__name__.lower()
        self.parent = None
        self.robot.robot0.set_pose(0,0,0)

        if not hasattr(self.robot, 'erouter'):
            self.robot.erouter = EventRouter()
            self.robot.erouter.robot = self.robot
            self.robot.erouter.start()
        else:
            self.robot.erouter.clear()

        self.launch_cam_viewer = launch_cam_viewer
        self.viewer = None
        self.annotate_sdk = annotate_sdk
        self.force_annotation = force_annotation
        self.annotated_scale_factor = annotated_scale_factor
        self.viewer_crosshairs = viewer_crosshairs
        self.speech = speech
        self.num_particles = num_particles
        self.landmarks = landmarks
        self.sensor_model = sensor_model
        self.landmark_test = landmark_test
        self.launch_particle_viewer = launch_particle_viewer
        self.particle_viewer_scale = particle_viewer_scale
        self.launch_path_viewer = launch_path_viewer
        #self.picked_up_callback = self.robot_picked_up
        self.put_down_handler = self.robot_put_down

        self.aruco = aruco
        self.aruco_marker_size = aruco_marker_size
        if self.aruco:
            self.robot.aruco_detector = \
                RobotArucoDetector(self.robot, dictionary_name, aruco_marker_size, aruco_disabled_ids)

        if particle_filter:
            self.particle_filter = particle_filter
        else:
            self.particle_filter = \
                SLAMParticleFilter(self.robot, landmark_test=self.landmark_test)

        self.perched_cameras = perched_cameras
        if self.perched_cameras:
            self.robot.perched = PerchedCameraThread(self.robot)

        self.robot.aruco_id = -1
        self.robot.use_shared_map = False
        #self.robot.world_map.server = ServerThread(self.robot)
        #self.robot.world_map.client = ClientThread(self.robot)
        #self.robot.world_map.is_server = True # Writes directly into perched.camera_pool

        self.launch_worldmap_viewer = launch_worldmap_viewer

    def start(self):
        global running_fsm
        running_fsm = self
        # Create a particle filter
        if self.particle_filter is None:
            self.particle_filter = ParticleFilter(self.robot,
                                                  num_particles=self.num_particles,
                                                  landmarks=self.landmarks,
                                                  sensor_model=self.sensor_model)
        # elif isinstance(self.particle_filter,SLAMParticleFilter):
        #    self.particle_filter.clear_landmarks()
        self.robot.particle_filter = self.particle_filter

        # Set up robot state
        self.robot.was_picked_up = False
        self.robot.carrying = None
        self.robot.fetching = None
        self.robot.robot0.set_light_color(vex.LightType.ALL, vex.Color.TRANSPARENT)
        self.robot.clear_actuators()

        # World map and path planner
        #self.robot.world.rrt = self.rrt or RRT(self.robot)

        # Polling
        self.set_polling_interval(0.025)  # for kine and motion model update

        # Launch viewers
        if self.launch_cam_viewer:
            if not self.robot.cam_viewer:
                self.robot.cam_viewer = \
                    CamViewer(self.robot, user_annotate_function=self.user_annotate)
                self.robot.cam_viewer.start()

        if self.launch_worldmap_viewer:
            if not self.robot.worldmap_viewer is True:
                self.robot.worldmap_viewer = WorldMapViewer(self.robot)
                self.robot.worldmap_viewer.start()

        if self.launch_particle_viewer:
            if not self.robot.particle_viewer:
                self.robot.particle_viewer = \
                    ParticleViewer(self.robot, scale=self.particle_viewer_scale)
                self.robot.particle_viewer.start()

        if self.launch_path_viewer:
            if not self.robot.path_viewer:
                self.robot.path_viewer = PathViewer(self.robot, self.robot.rrt)
            self.robot.path_viewer.start()

        if self.speech:
            self.robot.speech_listener.enable()
        else:
            self.robot.speech_listener.disable()

        # Call parent's start() to launch the state machine by invoking the start node.
        super().start()

    def stop(self):
        self.stop_children()
        super().stop()
        self.robot.erouter.clear()

    def stop_children(self):
        for node in self.children.values():
            node.stop()

    def poll(self):
        # Update robot kinematic description
        #self.robot.kine.get_pose()

        # Handle robot being picked up or put down
        if self.robot.is_picked_up():
            if not self.robot.was_picked_up:
                self.robot.robot0.stop_all_movement()
                self.robot.robot0.play_sound(vex.SoundType.HUAH, 50)
                self.robot.particle_filter.delocalize()
                self.robot.was_picked_up = True
        elif self.robot.was_picked_up:
            self.robot.was_picked_up = False
            self.robot.robot0.inertial.calibrate()
            self.robot.robot0.play_sound(vex.SoundType.DOORBELL, 50)
            self.robot.set_pose(0,0,0,0,reset_particles=False)
            self.put_down_handler()

        if not self.robot.was_picked_up:
            self.robot.particle_filter.move()
            self.robot.particle_filter.look_for_new_landmarks()
                
    def robot_put_down(self):
        pass

    def user_image(self,image,gray): pass

    def user_annotate(self,image):
        return image

    def process_image(self,image):
        # Aruco image processing
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        if self.aruco:
            self.robot.aruco_detector.process_image(gray)
        # Other image processors can run here if the user supplies them.
        self.user_image(image,gray)

################

def runfsm(module_name, running_modules=dict()):
    """runfsm('modname') reloads that module and expects it to contain
    a class of the same name. It calls that class's constructor and then
    calls the instance's start() method."""

    global running_fsm
    if running_fsm:
        running_fsm.stop()

    r_py = re.compile('.*\\.py$')
    if r_py.match(module_name):
        print("\n'%s' is not a module name. Trying '%s' instead.\n" %
              (module_name, module_name[0:-3]))
        module_name = module_name[0:-3]

    found = False
    try:
        reload(running_modules[module_name])
        found = True
    except KeyError: pass
    except: raise
    if not found:
        try:
            running_modules[module_name] = __import__(module_name)
        except ImportError as e:
            print("Error loading %s: %s.  Check your search path.\n" %
                  (module_name,e))
            return
        except Exception as e:
            print('\n===> Error loading %s:' % module_name)
            raise

    py_filepath = running_modules[module_name].__file__
    fsm_filepath = py_filepath[0:-2] + 'fsm'
    try:
        py_time = datetime.datetime.fromtimestamp(os.path.getmtime(py_filepath))
        fsm_time = datetime.datetime.fromtimestamp(os.path.getmtime(fsm_filepath))
        if py_time < fsm_time:
            cprint('Warning: %s.py is older than %s.fsm. Should you run genfsm?' %
                   (module_name,module_name), color="yellow")
    except: pass

    # The parent node class's constructor must match the module name.
    the_module = running_modules[module_name]
    the_class = the_module.__getattribute__(module_name) \
                if module_name in dir(the_module) else None
    if isinstance(the_class,type) and issubclass(the_class,StateNode) and not issubclass(the_class,StateMachineProgram):
        cprint("%s is not an instance of StateMachineProgram.\n" % module_name, color="red")
        return
    if not isinstance(the_class,type) or not issubclass(the_class,StateMachineProgram):
        cprint("Module %s does not contain a StateMachineProgram named %s.\n" %
              (module_name, module_name), color="red")
        return
    the_module.robot = evbase.robot_for_loading
    # Class's __init__ method will call setup, which can reference the above variables.
    running_fsm = the_class()
    evbase.robot_for_loading.loop.call_soon_threadsafe(running_fsm.start)
    return running_fsm

