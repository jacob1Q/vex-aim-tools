from .geometry import wrap_angle
import numpy as np

class KalmanFilter:
    def __init__(self, initial_estimate, initial_uncertainty, base_measurement_noise, process_noise):
        # Initialize state
        self.estimate = initial_estimate
        self.uncertainty = initial_uncertainty

        # Kalman filter parameters
        self.base_measurement_noise = base_measurement_noise
        self.measurement_noise = base_measurement_noise  # R
        self.process_noise = process_noise          # Q

        self.angle_history = []

    def update(self, measurement, noise=0):
        # Prediction step (no process dynamics, so state remains the same)
        predicted_estimate = self.estimate
        predicted_uncertainty = self.uncertainty + self.process_noise
        self.measurement_noise = self.base_measurement_noise + noise 

        # Kalman gain
        kalman_gain = predicted_uncertainty / (predicted_uncertainty + self.measurement_noise)

        # Update step
        self.estimate = predicted_estimate + kalman_gain * (measurement - predicted_estimate)
        self.uncertainty = (1 - kalman_gain) * predicted_uncertainty

        return self.estimate, self.uncertainty

    def update_circular(self, measurement):
        predicted_estimate = self.estimate
        z_estimate = np.array([np.sin(predicted_estimate), np.cos(predicted_estimate)])
        z_measure = np.array([np.sin(measurement), np.cos(measurement)])
        predicted_uncertainty = self.uncertainty + self.process_noise

        # Normalize measurement residual
        residual = z_measure - z_estimate

        # Kalman gain
        kalman_gain = predicted_uncertainty / (predicted_uncertainty + self.measurement_noise)

        # Update step
        self.estimate = predicted_estimate + kalman_gain * residual
        self.estimate = np.arctan2(self.estimate[0], self.estimate[1])

        self.uncertainty = (1 - kalman_gain) * predicted_uncertainty

        # Store estimate in history
        self.angle_history.append(self.estimate)

        # Compute circular variance
        circ_var = self.circular_variance()

        return self.estimate, self.uncertainty
    
    def circular_variance(self):
        if len(self.angle_history) == 0:
            return 0

        angles = np.array(self.angle_history)
        # Compute mean angle
        mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))

        # Compute circular variance
        circ_var = 1 - (np.sqrt(np.mean(np.sin(angles - mean_angle)**2) + np.mean(np.cos(angles - mean_angle)**2)) / 2)

        return circ_var


def neaten(x):
    if isinstance(x, (int,float)):
        return round(x*10)/10
    else:
        return x

class Pose():
    def __init__(self, x=0, y=0, z=0, theta=None, origin_id=-1):
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta
        self.origin_id = origin_id

    def __repr__(self):
        return f'<Pose x={neaten(self.x)} y={neaten(self.y)} z={neaten(self.z)} theta={neaten(self.theta)} origin_id={self.origin_id}>'

    def __sub__(self, other):
        angdiff = wrap_angle(self.theta - other.theta) if self.theta is not None and other.theta is not None else None
        return Pose(self.x - other.x,
                    self.y - other.y,
                    self.z - other.z,
                    angdiff)
    
    def is_comparable(self, other):
        return self.origin_id == other.origin_id
    

class PoseEstimate(Pose):
    def __init__(self, x=0, y=0, z=0, theta=None):
        if isinstance(x, Pose):
            p = x
            x = p.x
            y = p.y
            z = p.z
            theta = p.theta
        super().__init__(x, y, z, theta)
        initial_uncertainty = 200
        base_measurement_noise = 0.1
        process_noise = 0.01
        self.kf_x = KalmanFilter(x, initial_uncertainty, base_measurement_noise, process_noise)
        self.kf_y = KalmanFilter(y, initial_uncertainty, base_measurement_noise, process_noise)
        self.kf_z = KalmanFilter(z, initial_uncertainty, base_measurement_noise, process_noise)
        theta_initial_uncertainty = 200
        theta_base_measurement_noise = 0.5
        theta_process_noise = 0.05
        if theta is not None:
            self.kf_theta = KalmanFilter(theta,
                                         theta_initial_uncertainty,
                                         theta_base_measurement_noise,
                                         theta_process_noise)

    def update(self, new_pose, measurement_noise):
        self.x, _ = self.kf_x.update(new_pose.x, measurement_noise)
        self.y, _ = self.kf_y.update(new_pose.y, measurement_noise)
        self.z, _ = self.kf_z.update(new_pose.z, measurement_noise)
        if self.theta is not None:
            self.theta, _ = self.kf_theta.update_circular(new_pose.theta)

    def __repr__(self):
        return f'<PoseEstimate x={neaten(self.x)} y={neaten(self.y)} z={neaten(self.z)} theta={neaten(self.theta)} origin_id={self.origin_id}>'

