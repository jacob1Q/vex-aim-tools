"""
Create a dummy robot so we can use vex-aim-tools classes without
having to connect to a real robot.
"""

import asyncio

from .aim_kin import AIMKinematics
from .evbase import EventRouter
from .particle import SLAMParticleFilter
from .rrt import RRT
from .utils import Pose
from .worldmap import WorldMap

class SimRobot0():
    class Inertial():
        def get_heading(self): return 0

    def __init__(self):
        self.inertial = self.Inertial()
    def get_x_position(self): return 0
    def get_y_position(self): return 0

class SimRobot():
    def __init__(self, run_in_cloud=False):
        robot = self
        robot.loop = asyncio.get_event_loop()
        robot.robot0 = SimRobot0()

        robot.pose = Pose(0, 0, 0, 0)

        if not run_in_cloud:
            robot.erouter = EventRouter(robot)
            robot.holding = None

        robot.world_map = WorldMap(robot)
        robot.particle_filter = SLAMParticleFilter(robot)
        robot.kine = AIMKinematics(robot)
        robot.rrt = RRT(robot)

