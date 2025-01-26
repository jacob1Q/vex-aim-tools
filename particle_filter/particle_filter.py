import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import tkinter as tk
from tkinter import simpledialog
import aim_fsm
from aim_fsm import *

class ParticleFilter:
    def __init__(self, num_particles, landmarks, init_pose):
        self.num_particles = num_particles
        self.particles = np.array([init_pose + np.random.randn(3)*0.5 for _ in range(num_particles)])
        self.weights = np.ones(num_particles) / num_particles
        self.landmarks = landmarks

    def motion_update(self, control, noise):
        for i, p in enumerate(self.particles):
            dx = control[0] + np.random.randn() * noise[0]
            dy = control[1] + np.random.randn() * noise[1]
            dtheta = control[2] + np.random.randn() * noise[2]
            self.particles[i] += [dx, dy, dtheta]

    def measurement_update(self, measurements):
        for i, p in enumerate(self.particles):
            weight = 1.0
            for lm_id, lm_meas in measurements.items():
                expected = self.expected_measurement(p, self.landmarks[lm_id])
                weight *= self.gaussian(expected, lm_meas, 1.0)
            self.weights[i] = weight
        self.weights += 1e-300
        self.weights /= sum(self.weights)

    def resample(self):
        indices = np.random.choice(
            range(self.num_particles),
            size=self.num_particles,
            p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def expected_measurement(self, particle, landmark):
        dx, dy = landmark[0] - particle[0], landmark[1] - particle[1]
        return np.sqrt(dx**2 + dy**2)

    def gaussian(self, expected, observed, std):
        return np.exp(-((expected - observed)**2) / (2 * std**2)) / (np.sqrt(2 * np.pi) * std)

    def estimate(self):
        return np.average(self.particles, weights=self.weights, axis=0)

class RobotLocalizationGUI:
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.robot_pose = [0, 0, 0]
        self.pf = ParticleFilter(1000, landmarks, self.robot_pose)
        self.root = tk.Tk()
        self.root.geometry("600x600")
        self.canvas = tk.Canvas(self.root, width=600, height=600, bg="white")
        self.canvas.pack()
        self.draw_landmarks()

    def draw_landmarks(self):
        for lm_id, lm in self.landmarks.items():
            self.canvas.create_oval(
                lm[0]-5+300, lm[1]-5+300, lm[0]+5+300, lm[1]+5+300,
                fill="blue"
            )
            self.canvas.create_text(lm[0]+300, lm[1]+300, text=f"{lm_id}")

    def draw_particles(self):
        self.canvas.delete("particles")
        for p in self.pf.particles:
            self.canvas.create_oval(
                p[0]-2+300, p[1]-2+300, p[0]+2+300, p[1]+2+300,
                fill="red", tags="particles"
            )

    def draw_robot(self):
        self.canvas.delete("robot")
        x, y, _ = self.robot_pose
        self.canvas.create_oval(
            x-10+300, y-10+300, x+10+300, y+10+300,
            fill="green", tags="robot"
        )

    def update_pose(self, pose):
        self.robot_pose = pose

    def update_particles(self, control):
        self.pf.motion_update(control, [0.1, 0.1, 0.05])
        measurements = {lm_id: self.get_measurement(self.robot_pose, lm) for lm_id, lm in self.landmarks.items()}
        self.pf.measurement_update(measurements)
        self.pf.resample()

    def get_measurement(self, pose, landmark):
        dx, dy = landmark[0] - pose[0], landmark[1] - pose[1]
        return np.sqrt(dx**2 + dy**2)

    def run(self):
        def update():
            control = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.0]
            self.update_pose([self.robot_pose[0]+control[0], self.robot_pose[1]+control[1], 0])
            self.update_particles(control)
            self.draw_particles()
            self.draw_robot()
            self.root.after(100, update)

        update()
        self.root.mainloop()

class particle_filter(StateMachineProgram):    
    def setup(self):
        landmarks = {
            1: [130, 5],
            2: [136, -40]
        }
        gui = RobotLocalizationGUI(landmarks)
        gui.run()
'''
C> Added <ArucoMarkerObj 38: (130.4, 5.0, 20.0) @ -167 deg. visible>
Added <ArucoMarkerObj 39: (136.5, -40.8, 20.0) @ -176 deg. visible>
'''