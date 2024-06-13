import numpy as np

ENV_COLL_THRESH = 0.05 # up to 5% of vehicle can be off the road
VEH_COLL_THRESH = 0.02 # IoU must be over this to count as a collision for metric (not loss)

def check_single_veh_coll(traj_tgt, lw_tgt, traj_others, lw_others):
    '''
    Checks if the target trajectory collides with each of the given other trajectories.

    Assumes all trajectories and attributes are UNNORMALIZED. Handles nan frames in traj_others by simply skipping.

    :param traj_tgt: (T x 4) with T the nb of time_steps and 4 the state space
    -> the state space : (coord x of center, coord y of center, velocity x = cos of heading, velocity y = sin of heading)
    :param lw_tgt: (2, ) with length and width of the target vehicle
    :param traj_others: (N x T x 4) with N the number of other vehicles/agents, T the nb of time_steps and 4 the state space
    :param lw_others: (N x 2) with N the number of other vehicles, associated to length and width of the other vehicles

    :returns veh_coll: (N)
    :returns coll_time: (N)
    '''

    from shapely.geometry import Polygon

    NA, FT, _ = traj_others.shape # number of agents, number of frames, state space dim (discarded)
    # traj_tgt = traj_tgt.cpu().numpy() # convert to numpy
    # lw_tgt = lw_tgt.cpu().numpy() # convert to numpy
    # traj_others = traj_others.cpu().numpy() # convert to numpy
    # lw_others = lw_others.cpu().numpy() # convert to numpy

    veh_coll = np.zeros((NA), dtype=np.bool_) # whether each of the N vehicley collides with the target (boolean)
    coll_time = np.ones((NA), dtype=np.int_)*FT # time of collision for each of the N vehicles (initialized to FT, the last time_step)
    poly_cache = dict() # for the tgt polygons since used many times
    for aj in range(NA):
        for t in range(FT):
            # compute iou (Intersection over Union) between target and other vehicle at time t
            if t not in poly_cache: # executed only on first iteration of outer loop
                ai_state = traj_tgt[t, :]
                ai_corners = get_corners(ai_state, lw_tgt)
                ai_poly = Polygon(ai_corners)
                poly_cache[t] = ai_poly
            else:
                ai_poly = poly_cache[t]

            aj_state = traj_others[aj, t, :]
            if np.sum(np.isnan(aj_state)) > 0:
                continue
            aj_corners = get_corners(aj_state, lw_others[aj])
            aj_poly = Polygon(aj_corners)
            cur_iou = ai_poly.intersection(aj_poly).area / ai_poly.union(aj_poly).area
            if cur_iou > VEH_COLL_THRESH:
                veh_coll[aj] = True
                coll_time[aj] = t
                break # don't need to check rest of sequence

    return veh_coll, coll_time
 

def get_corners(box, lw):
    l, w = lw
    simple_box = np.array([
        [-l/2., -w/2.],
        [l/2., -w/2.],
        [l/2., w/2.],
        [-l/2., w/2.],
    ])
    h = np.arctan2(box[3], box[2])
    rot = get_rot(h)
    simple_box = np.dot(simple_box, rot)
    simple_box += box[:2] # rotated box translated to its correct position by adding the center coordinates to each of the corners

    return simple_box

def get_rot(h):
    return np.array([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])