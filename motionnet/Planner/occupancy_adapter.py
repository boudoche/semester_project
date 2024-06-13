import scipy
import numpy as np
from common_utils import *
from nuplan.planning.metrics.utils.expert_comparisons import principal_value

# min number of points in ref path should be equal to 120, so we fix MAX_LEN = 12
MAX_LEN = 12

# number of time_steps should be equal to 60, so we fix T = 6 here
T = 6
def occupancy_adpter(predictions, scores, neighbors, ref_path):

    """
            predictions: (nb_vehicles, possible_modes, nb_time_steps=60, nb_attributes) w/ first two attributes x and y coords
            scores: (nb_vehicles, possible_modes)
            neighbors: (nb_neighbors, nb_attributes=8)
            ref_path: (path_len=nb_points, nb_attributes=2) 
    """

        # determines index of the most probable mode for each vehicle's predicted trajectory : 1D array of shape (nb_vehicles,)
        # best_mode = np.argmax(scores.cpu().numpy(), axis=-1)
    best_mode = np.argmax(scores, axis=-1)
            
        # predictions = predictions.cpu().numpy()
        #neighbors = neighbors.cpu().numpy()
    
        # extracts the best (most probable) trajectory for each vehicle
    best_predictions = [predictions[i, best_mode[i], :, :2] for i in range(predictions.shape[0])]
        # best_predictions is a list where each element is a 2D array representing the x and y coordinates of the best
        # trajectory for a specific vehicle across all time steps.
            
        # converts each best trajectory from Cartesian coordinates to the Frenet coordinate system relative to the ref path
    prediction_F = [transform_to_Frenet(a, ref_path) for a in best_predictions]
        # prediction_F is a list where each element contains the Frenet coordinates (s and l, and optionally heading h)
        # for each best trajectory.

    len_path = ref_path.shape[0] # length of the reference path
    if len_path < MAX_LEN * 10:
                # if the ref path is shorter than the minimum path length, we extend it by repeating the last point
        ref_path = np.append(ref_path, np.repeat(ref_path[np.newaxis, -1], MAX_LEN*10 -len(ref_path), axis=0), axis=0)
    
    time_occupancy = np.stack(T * 10 * [ref_path[:, -1]], axis=0) # (timestep, path_len)

        # fills the time_occupancy array
    for t in range(T * 10): # iterates over the time steps 
        for n, a in enumerate(prediction_F):
                # Skip if the neighbor is invalid or not considered
            if neighbors[n][0] == 0: 
                continue
                # Skip if the Frenet `s` coordinate is not positive
            if a[0][0] <= 0:
                continue
            
                # Calculate intersection threshold
            aw = neighbors[n][7]
            threshold = aw * 0.5 + WIDTH * 0.5 + 0.3

                    # Check if Frenet coordinates of vehicle a fall within the threshold
            if a[t][0] > 0 and np.abs(a[t][1]) < threshold:
                al = neighbors[n][6]
                backward = 0.5 * al + 3
                forward = 0.5 * al
                os = np.clip(a[t][0] - backward, 0, MAX_LEN)
                oe = np.clip(a[t][0] + forward, 0, MAX_LEN)
                time_occupancy[t][int(os*10):int(oe*10)] = 1

                # Extend occupancy if path length is shorter than minimum path length
        if len_path < MAX_LEN * 10:
            time_occupancy[t][len_path:] = 1

    time_occupancy = np.reshape(time_occupancy, (T*10, -1, 10))
    time_occupancy = np.max(time_occupancy, axis=-1)

    return time_occupancy


def transform_to_Frenet(traj, ref_path):
    distance_to_ref_path = scipy.spatial.distance.cdist(traj[:, :2], ref_path[:, :2])
    frenet_idx = np.argmin(distance_to_ref_path, axis=-1)
    ref_points = ref_path[frenet_idx]
    interval = 0.1

    frenet_s = interval * frenet_idx
    e = np.sign((traj[:, 1] - ref_points[:, 1]) * np.cos(ref_points[:, 2]) - (traj[:, 0] - ref_points[:, 0]) * np.sin(ref_points[:, 2]))
    frenet_l = np.linalg.norm(traj[:, :2] - ref_points[:, :2], axis=-1) * e 

    if traj.shape[-1] == 3:
        frenet_h = principal_value(ref_points[:, 2] - traj[:, 2])
        frenet_traj = np.column_stack([frenet_s, frenet_l, frenet_h])
    else:
        frenet_traj = np.column_stack([frenet_s, frenet_l])

    return frenet_traj