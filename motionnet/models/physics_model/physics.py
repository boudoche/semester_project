import abc
from typing import Tuple

import numpy as np

KinematicsData = Tuple[float, float, float, float, float, float, float, float, float, float]


def _kinematics_from_tokens(obj_trajs) -> KinematicsData:
    """
    Returns the 2D position, velocity and acceleration vectors from the given track records,
    along with the speed, yaw rate, (scalar) acceleration (magnitude), and heading.
    :param helper: Instance of PredictHelper.
    :instance: Token of instance.
    :sample: Token of sample.
    :return: KinematicsData.
    """

    x = obj_trajs[0, 20, 0]
    y = obj_trajs[0, 20, 1]

    hx = obj_trajs[0, 20, 34]
    hy = obj_trajs[0, 20, 33]
    vx = obj_trajs[0, 20, 35]
    vy = obj_trajs[0, 20, 36]
    ax = obj_trajs[0, 20, 37]
    ay = obj_trajs[0, 20, 38]

    velocity = np.sqrt(vx**2 + vy**2)
    yaw = np.arctan2(hy, hx)
    yaw_rate = np.arctan2(ay, ax)
    acceleration = np.sqrt(ax**2 + ay**2)

    return x, y, vx, vy, ax, ay, velocity, yaw_rate, acceleration, yaw
    # OK


def _constant_velocity_heading_from_kinematics(kinematics_data: KinematicsData,
                                               sec_from_now: float,
                                               sampled_at: int) -> np.ndarray:
    """
    Computes a constant velocity baseline for given kinematics data, time window
    and frequency.
    :param kinematics_data: KinematicsData for agent.
    :param sec_from_now: How many future seconds to use.
    :param sampled_at: Number of predictions to make per second.
    """
    x, y, vx, vy, _, _, _, _, _, _ = kinematics_data
    preds = []
    time_step = 1.0 / sampled_at
    for time in np.arange(time_step, sec_from_now + time_step, time_step):
        preds.append((x + time * vx, y + time * vy, vx, vy))
    return np.array(preds)
    # OK

def _constant_velocity_heading_from_kinematics(kinematics_data: KinematicsData,
                                               sec_from_now: float,
                                               sampled_at: int) -> np.ndarray:
    """
    Computes a constant velocity baseline for given kinematics data, time window
    and frequency.
    :param kinematics_data: KinematicsData for agent.
    :param sec_from_now: How many future seconds to use.
    :param sampled_at: Number of predictions to make per second.
    """
    x, y, vx, vy, _, _, _, _, _, _ = kinematics_data
    preds = []
    time_step = 1.0 / sampled_at
    for time in np.arange(time_step, sec_from_now + time_step, time_step):
        preds.append((x + time * vx, y + time * vy, vx, vy))
    return np.array(preds)
    # OK


def _constant_acceleration_and_heading(kinematics_data: KinematicsData,
                                       sec_from_now: float, sampled_at: int) -> np.ndarray:
    """
    Computes a baseline prediction for the given time window and frequency, under
    the assumption that the acceleration and heading are constant.
    :param kinematics_data: KinematicsData for agent.
    :param sec_from_now: How many future seconds to use.
    :param sampled_at: Number of predictions to make per second.
    """
    x, y, vx, vy, ax, ay, _, _, _, _ = kinematics_data

    preds = []
    time_step = 1.0 / sampled_at
    vxv = vx
    vyv = vy
    for time in np.arange(time_step, sec_from_now + time_step, time_step):
        half_time_squared = 0.5 * time * time
        preds.append((x + time * vx + half_time_squared * ax,
                      y + time * vy + half_time_squared * ay, vxv, vyv))
        vxv += ax * time_step
        vyv += ay * time_step
    return np.array(preds)
    # OK


def _constant_speed_and_decreasing_yaw_rate(kinematics_data: KinematicsData,
                                 sec_from_now: float, sampled_at: int, decreasing_ratio : float) -> np.ndarray:
    """
    Computes a baseline prediction for the given time window and frequency, under
    the assumption that the (scalar) speed and yaw rate are constant.
    :param kinematics_data: KinematicsData for agent.
    :param sec_from_now: How many future seconds to use.
    :param sampled_at: Number of predictions to make per second.
    """
    x, y, vx, vy, _, _, velocity, yaw_rate, _, yaw = kinematics_data

    preds = []
    time_step = 1.0 / sampled_at
    distance_step = time_step * velocity
    yaw_step = time_step * yaw_rate/decreasing_ratio
    for _ in np.arange(time_step, sec_from_now + time_step, time_step):
        x += distance_step * np.cos(yaw)
        y += distance_step * np.sin(yaw)
        preds.append((x, y, vx, vy))
        yaw += yaw_step
    return np.array(preds)
    # OK





def _constant_acceleration_and_heading(kinematics_data: KinematicsData,
                                       sec_from_now: float, sampled_at: int) -> np.ndarray:
    """
    Computes a baseline prediction for the given time window and frequency, under
    the assumption that the acceleration and heading are constant.
    :param kinematics_data: KinematicsData for agent.
    :param sec_from_now: How many future seconds to use.
    :param sampled_at: Number of predictions to make per second.
    """
    x, y, vx, vy, ax, ay, _, _, _, _ = kinematics_data

    preds = []
    time_step = 1.0 / sampled_at
    vxv = vx
    vyv = vy
    for time in np.arange(time_step, sec_from_now + time_step, time_step):
        half_time_squared = 0.5 * time * time
        preds.append((x + time * vx + half_time_squared * ax,
                      y + time * vy + half_time_squared * ay, vxv, vyv))
        vxv += ax * time_step
        vyv += ay * time_step
    return np.array(preds)
    # OK


def _constant_speed_and_yaw_rate(kinematics_data: KinematicsData,
                                 sec_from_now: float, sampled_at: int) -> np.ndarray:
    """
    Computes a baseline prediction for the given time window and frequency, under
    the assumption that the (scalar) speed and yaw rate are constant.
    :param kinematics_data: KinematicsData for agent.
    :param sec_from_now: How many future seconds to use.
    :param sampled_at: Number of predictions to make per second.
    """
    x, y, vx, vy, _, _, velocity, yaw_rate, _, yaw = kinematics_data

    preds = []
    time_step = 1.0 / sampled_at
    distance_step = time_step * velocity
    yaw_step = time_step * yaw_rate/10
    for _ in np.arange(time_step, sec_from_now + time_step, time_step):
        x += distance_step * np.cos(yaw)
        y += distance_step * np.sin(yaw)
        preds.append((x, y, vx, vy))
        yaw += yaw_step
    return np.array(preds)
    # OK


def _constant_magnitude_accel_and_yaw_rate(kinematics_data: KinematicsData,
                                           sec_from_now: float, sampled_at: int) -> np.ndarray:
    """
    Computes a baseline prediction for the given time window and frequency, under
    the assumption that the rates of change of speed and yaw are constant.
    :param kinematics_data: KinematicsData for agent.
    :param sec_from_now: How many future seconds to use.
    :param sampled_at: Number of predictions to make per second.
    """
    x, y, vx, vy, _, _, velocity, yaw_rate, accel, yaw = kinematics_data

    preds = []
    time_step = 1.0 / sampled_at
    speed_step = time_step * accel
    yaw_step = time_step * yaw_rate/10
    vxv = vx
    vyv = vy
    for _ in np.arange(time_step, sec_from_now + time_step, time_step):
        distance_step = time_step * velocity
        x += distance_step * np.cos(yaw)
        y += distance_step * np.sin(yaw)
        preds.append((x, y, vxv, vyv))
        velocity += speed_step
        yaw += yaw_step
        vxv = velocity * np.cos(yaw)
        vyv = velocity * np.sin(yaw)
    return np.array(preds)
    # OK

# provides a common interface and shared functionality for different types of trajectory prediction models (baselines)
class Baseline(abc.ABC):

    def __init__(self, sec_from_now: float, obj_trajs: np.ndarray, ground_truth: np.ndarray, sampled_at=10):
        """
        Inits Baseline.
        :param sec_from_now: How many seconds into the future to make the prediction.
        :param helper: Instance of PredictHelper.
        """
        assert sec_from_now % 0.5 == 0, f"Parameter sec from now must be divisible by 0.5. Received {sec_from_now}."
        self.obj_trajs = obj_trajs
        self.sec_from_now = sec_from_now
        self.ground_truth = ground_truth
        self.sampled_at = sampled_at

    @abc.abstractmethod
    def __call__(self) -> np.ndarray:
        pass


        


class PhysicsOracle(Baseline):
    """ Makes several physics-based predictions and picks the one closest to the ground truth. """

    def __call__(self) -> np.ndarray:
        """
        Makes prediction.
        :param token: string of format {instance_token}_{sample_token}.
        """
        kinematics = _kinematics_from_tokens(self.obj_trajs) # ok
        ground_truth = self.ground_truth

        assert ground_truth.shape[0] == int(self.sec_from_now * self.sampled_at), ("Ground truth does not correspond "
                                                                                   f"to {self.sec_from_now} seconds.")

        path_funs = [
            _constant_acceleration_and_heading,
            _constant_magnitude_accel_and_yaw_rate,
            _constant_speed_and_yaw_rate,
            _constant_velocity_heading_from_kinematics
        ]

        paths = [path_fun(kinematics, self.sec_from_now, self.sampled_at) for path_fun in path_funs]

        # Select the one with the least l2 error, averaged (or equivalently, summed) over all
        # points of the path.  This is (proportional to) the Frobenius norm of the difference
        # between the path (as an n x 2 matrix) and the ground truth.
        oracle = sorted(paths, key=lambda path: np.linalg.norm(np.array(path) - np.array(ground_truth), ord="fro"))[0]

        # Need the prediction to have 2d.
        # we need to return only x,y,vx,vy predited for each time step
        return oracle