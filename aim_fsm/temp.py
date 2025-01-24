import math
import random
import numpy as np
from math import pi, sqrt, sin, cos, atan2, exp
from .geometry import wrap_angle, wrap_selected_angles
from .aruco import ArucoMarker
from .worldmap import WorldObject, ArucoMarkerObj

class Particle:
    def __init__(self, index=-1):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.log_weight = 0
        self.weight = 1
        self.index = index

    def __repr__(self):
        return '<Particle %d: (%.2f, %.2f) %.1f deg. log_wt=%f>' % \
               (self.index, self.x, self.y, self.theta * 180 / pi, self.log_weight)

class SensorModel:
    def __init__(self, robot, landmarks=None):
        self.robot = robot
        if landmarks is None:
            landmarks = dict()
        self.set_landmarks(landmarks)

    def set_landmarks(self, landmarks):
        self.landmarks = landmarks

    def compute_robot_motion(self):
        # Placeholder for computing robot motion; integrate with VEX robot's odometry.
        dx = self.robot.x - self.robot.last_x
        dy = self.robot.y - self.robot.last_y
        dtheta = wrap_angle(self.robot.theta - self.robot.last_theta)
        self.robot.last_x = self.robot.x
        self.robot.last_y = self.robot.y
        self.robot.last_theta = self.robot.theta
        return dx, dy, dtheta

class ArucoDistanceSensorModel(SensorModel):
    def __init__(self, robot, landmarks=None, distance_variance=100):
        super().__init__(robot, landmarks)
        self.distance_variance = distance_variance

    def evaluate(self, particles):
        for particle in particles:
            for marker_id, landmark in self.landmarks.items():
                dx = landmark.x - particle.x
                dy = landmark.y - particle.y
                predicted_dist = sqrt(dx**2 + dy**2)
                error = self.robot.sensor_distance - predicted_dist
                particle.log_weight -= (error**2) / self.distance_variance

class ArucoBearingSensorModel(SensorModel):
    def __init__(self, robot, landmarks=None, bearing_variance=0.1):
        super().__init__(robot, landmarks)
        self.bearing_variance = bearing_variance

    def evaluate(self, particles):
        for particle in particles:
            for marker_id, landmark in self.landmarks.items():
                dx = landmark.x - particle.x
                dy = landmark.y - particle.y
                predicted_bearing = wrap_angle(atan2(dy, dx) - particle.theta)
                error = wrap_angle(self.robot.sensor_bearing - predicted_bearing)
                particle.log_weight -= (error**2) / self.bearing_variance

class ArucoCombinedSensorModel(SensorModel):
    def __init__(self, robot, landmarks=None, distance_variance=100, bearing_variance=0.1):
        super().__init__(robot, landmarks)
        self.distance_variance = distance_variance
        self.bearing_variance = bearing_variance

    def evaluate(self, particles):
        for particle in particles:
            for marker_id, landmark in self.landmarks.items():
                dx = landmark.x - particle.x
                dy = landmark.y - particle.y

                # Distance
                predicted_dist = sqrt(dx**2 + dy**2)
                dist_error = self.robot.sensor_distance - predicted_dist
                particle.log_weight -= (dist_error**2) / self.distance_variance

                # Bearing
                predicted_bearing = wrap_angle(atan2(dy, dx) - particle.theta)
                bearing_error = wrap_angle(self.robot.sensor_bearing - predicted_bearing)
                particle.log_weight -= (bearing_error**2) / self.bearing_variance

class ParticleFilter:
    def __init__(self, robot, num_particles=100, landmarks=None):
        self.robot = robot
        self.num_particles = num_particles
        self.particles = [Particle(i) for i in range(num_particles)]
        self.sensor_model = ArucoCombinedSensorModel(robot, landmarks)

    def predict(self, control, noise):
        for particle in self.particles:
            particle.x += control[0] + random.gauss(0, noise[0])
            particle.y += control[1] + random.gauss(0, noise[1])
            particle.theta += control[2] + random.gauss(0, noise[2])

    def update(self):
        self.sensor_model.evaluate(self.particles)
        max_log_weight = max(p.log_weight for p in self.particles)
        for particle in self.particles:
            particle.weight = exp(particle.log_weight - max_log_weight)

    def resample(self):
        weights = [particle.weight for particle in self.particles]
        indices = random.choices(range(self.num_particles), weights, k=self.num_particles)
        self.particles = [self.particles[i] for i in indices]

    def get_pose_estimate(self):
        x = sum(p.x * p.weight for p in self.particles) / sum(p.weight for p in self.particles)
        y = sum(p.y * p.weight for p in self.particles) / sum(p.weight for p in self.particles)
        theta = sum(p.theta * p.weight for p in self.particles) / sum(p.weight for p in self.particles)
        return x, y, theta

