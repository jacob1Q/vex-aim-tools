################ Imports
import numpy as np
import asyncio
import atexit
import code
import datetime
import logging
import os
import platform
import re
import readline
import rlcompleter
import subprocess
import sys
import threading
import time
import traceback
from importlib import __import__, reload

try:
    from termcolor import cprint
except:
    def cprint(string, color=None):
        print(string)


import matplotlib
import matplotlib.pyplot as plt


import aim_fsm
from aim_fsm import *
class KF_v1(StateMachineProgram):    
    def setup(self):
        while True:
            seen_marker_objects = self.robot.world_map.objects #all objects
            seen_marker_objects = {key: value for key, value in seen_marker_objects.items() if "Aruco" in key} #only aruco markers

            for (id, marker) in seen_marker_objects.items():
                # if marker.id_string in self.landmarks:
                #     sensor_dist = None
                    # landmark_spec = self.landmarks[marker.id_string] #needs some tweaking to satisfy the type of landmark_spec
                    # lm_x = landmark_spec.position.x
                    # lm_y = landmark_spec.position.y

                print('hhhhhh')
                print(id)
                print(marker.pose.x)


 
        print('ffff')
    
    # def setup(self):
    #     initial_estimate = 0
    #     initial_uncertainty = 200
    #     measurement_noise = 0.25
    #     process_noise = 0.25
    #     self.robot.set_pose(0,0,0)
    #     objs = sorted(self.robot.world_map.objects.items(), key=lambda x: x[0])
    #     if len(objs) == 0:
    #         print('No objects in the world map.\n')
    #         return
    #     kf_x=self.KalmanFilter(initial_estimate=0, initial_uncertainty=200, measurement_noise=0.1, process_noise=0.01)
    #     kf_y=self.KalmanFilter(initial_estimate, initial_uncertainty, measurement_noise=0.1, process_noise=0.01)
    #     x_coord,y_coord=[],[]
    #     estimates_x,estimates_y = [],[]
    #     uncertainties_x,uncertainties_y = [],[]
    #     i=0
    #     while True:
    #         print('Objects in the world map:')
            
    #         for obj in objs:
    #             if obj[1].is_visible and 'OrangeBarrel' in obj[0]:
    #                 i=i+1
    #                 print(f'{obj[0]}: {obj[1]}')
    #                 print(f'  {obj[1].x} {obj[1].y}')
    #                 x_coord.append(obj[1].x)
    #                 y_coord.append(obj[1].y)
    #                 estimate_x, uncertainty_x = kf_x.update(obj[1].x)
    #                 estimate_y, uncertainty_y = kf_y.update(obj[1].y)
    #                 estimates_x.append(estimate_x)
    #                 uncertainties_y.append(uncertainty_y)
    #                 estimates_y.append(estimate_y)
    #                 uncertainties_x.append(uncertainty_x)
    #                 time.sleep(0.1)
        
    #         print(f'measurement {i} taken')
    #         if i%100==0:
    #             self.visualize_uncertainty(x_coord, estimates_x, uncertainties_x)
    #             self.visualize_uncertainty(y_coord, estimates_y, uncertainties_y)
    #             print('Uncertainty reduced')
    #             time.sleep(1)

    #         if i==100:
    #             break
                
            
    #     print('ffff')

    def filter(self):
        # Parameters
        initial_estimate = 0
        initial_uncertainty = 10
        measurement_noise = 4
        process_noise = 1

        # Simulated measurements (with noise)
        true_position = 50
        num_measurements = 20
        measurements = [true_position + np.random.normal(0, np.sqrt(measurement_noise)) for _ in range(num_measurements)]

        # Kalman filter initialization
        kf = KalmanFilter(initial_estimate, initial_uncertainty, measurement_noise, process_noise)

        # Kalman filter updates
        estimates = []
        uncertainties = []
        for measurement in measurements:
            estimate, uncertainty = kf.update(measurement)
            estimates.append(estimate)
            uncertainties.append(uncertainty)

        # Visualize the reduction in uncertainty
        visualize_uncertainty(measurements, estimates, uncertainties)


    class KalmanFilter:
        def __init__(self, initial_estimate, initial_uncertainty, measurement_noise, process_noise):
            # Initialize state
            self.estimate = initial_estimate
            self.uncertainty = initial_uncertainty

            # Kalman filter parameters
            self.measurement_noise = measurement_noise  # R
            self.process_noise = process_noise          # Q

        def update(self, measurement):
            # Prediction step (no process dynamics, so state remains the same)
            predicted_estimate = self.estimate
            predicted_uncertainty = self.uncertainty + self.process_noise

            # Kalman gain
            kalman_gain = predicted_uncertainty / (predicted_uncertainty + self.measurement_noise)

            # Update step
            self.estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
            self.uncertainty = (1 - kalman_gain) * predicted_uncertainty

            return self.estimate, self.uncertainty

    def visualize_uncertainty(self,measurements, estimates, uncertainties):
        iterations = np.arange(len(measurements))

        plt.figure(figsize=(10, 6))
        
        # Plot measurements
        plt.plot(iterations, measurements, label='Measurements', linestyle='dotted', color='red')

        # Plot estimates
        plt.plot(iterations, estimates, label='Estimates', linestyle='solid', color='blue')

        # Plot uncertainties
        plt.fill_between(iterations, 
                        np.array(estimates) - np.array(uncertainties), 
                        np.array(estimates) + np.array(uncertainties), 
                        color='blue', alpha=0.2, label='Uncertainty Range')

        plt.title('Kalman Filter: Reduction in Uncertainty Over Time')
        plt.xlabel('Measurement Iteration')
        plt.ylabel('Position')
        plt.legend()
        plt.grid()
        plt.show()



