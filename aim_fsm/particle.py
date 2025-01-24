"""
Particle filter localization.
"""

import math
import random
import numpy as np
from math import pi, sqrt, sin, cos, atan2, exp
from .geometry import wrap_angle, wrap_selected_angles
from .aruco import ArucoMarker
from .worldmap import WorldObject, ArucoMarkerObj

class Particle():
    def __init__(self, index=-1):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.log_weight = 0
        self.weight = 1
        self.index = index

    def __repr__(self):
        return '<Particle %d: (%.2f, %.2f) %.1f deg. log_wt=%f>' % \
               (self.index, self.x, self.y, self.theta*80/pi, self.log_weight)
    
#================ Particle Initializers ================

#to create a particle filter initialisation
class ParticleInitializer():
    def __init__(self):
        self.pf = None   # must be filled in after creation

#random initialisation of particles within a certain radius
class RandomWithinRadius(ParticleInitializer):
    """ Normally distribute particles within a radius, with random heading. """
    def __init__(self,radius=200):
        super().__init__()
        self.radius = radius

    def initialize(self, robot):
        for p in self.pf.particles:  #pf.particles - list of particles
            qangle = random.random()*2*pi
            r = random.gauss(0, self.radius/2) + self.radius/1.5
            p.x = r * cos(qangle)
            p.y = r * sin(qangle)
            p.theta = random.random()*2*pi
            p.log_weight = 0.0
            p.weight = 1.0
        self.pf.pose = (0, 0, 0)
        self.pf.motion_model.old_pose = robot.pose

class RobotPosition(ParticleInitializer):
    """ Initialize all particles to the robot's current position or a constant;
    the motion model will jitter them. """
    def __init__(self, x=None, y=None, theta=None):
        super().__init__()
        self.x = x
        self.y = y
        self.theta = theta

    def initialize(self, robot):
        if self.x is None:
            x = robot.pose.position.x
            y = robot.pose.position.y
            theta = robot.pose.rotation.angle_z.radians
        else:
            x = self.x
            y = self.y
            theta = self.theta
        for p in self.pf.particles:
            p.x = x
            p.y = y
            p.theta = theta
            p.log_weight = 0.0
            p.weight = 1.0
        self.pf.pose = (x, y, theta)
        self.pf.motion_model.old_pose = robot.pose
