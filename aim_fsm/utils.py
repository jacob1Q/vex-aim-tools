from .geometry import wrap_angle

class KalmanFilter:
    def __init__(self, initial_estimate, initial_uncertainty, base_measurement_noise, process_noise):
        # Initialize state
        self.estimate = initial_estimate
        self.uncertainty = initial_uncertainty

        # Kalman filter parameters
        self.base_measurement_noise = base_measurement_noise
        self.measurement_noise = base_measurement_noise  # R
        self.process_noise = process_noise          # Q

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
        return self.update(measurement) # **** TEMPORARY HACK ****

def neaten(x):
    if isinstance(x, (int,float)):
        return round(x*10)/10
    else:
        return x

class Pose():
    def __init__(self, x=0, y=0, z=0, theta=None):
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta

    def __repr__(self):
        return f'<Pose x={neaten(self.x)} y={neaten(self.y)} z={neaten(self.z)} theta={neaten(self.theta)}>'

    def __sub__(self, other):
        angdiff = wrap_angle(self.theta - other.theta) if self.theta is not None else None
        return Pose(self.x - other.x,
                    self.y - other.y,
                    self.z - other.z,
                    angdiff)


class PoseEstimate(Pose):
    def __init__(self, x=0, y=0, z=0, theta=None):
        if isinstance(x, Pose):
            p = x
            x = p.x
            y = p.y
            z = p.z
            theta = p.theta
        super().__init__(x, y, z, theta)
        self.kf_x = KalmanFilter(x, 200, 0.1, 0.01)
        self.kf_y = KalmanFilter(y, 200, 0.1, 0.01)
        self.kf_z = KalmanFilter(z, 200, 0.1, 0.01)
        if theta is not None:
            self.kf_theta = KalmanFilter(theta, 1, 0.5, 0.05)

    def update(self, new_pose, measurement_noise):
        self.x, _ = self.kf_x.update(new_pose.x, measurement_noise)
        self.y, _ = self.kf_y.update(new_pose.y, measurement_noise)
        self.z, _ = self.kf_z.update(new_pose.z, measurement_noise)
        if self.theta is not None:
            self.theta, _ = self.kf_theta.update_circular(new_pose.theta)

    def __repr__(self):
        return f'<PoseEstimate x={neaten(self.x)} y={neaten(self.y)} z={neaten(self.z)} theta={neaten(self.theta)}>'

