import scipy
import numpy as np
import matplotlib.pyplot as plt
from datasets.common_utils import *
from shapely import Point, LineString, Polygon
from shapely.geometry.base import CAP_STYLE
from .bezier_path import calc_4points_bezier_path
from .cubic_spline import calc_spline_course
import datasets.types as types


MAX_LEN = 120



class LatticePlanner: 
    def __init__(self, route_ids, max_len=120):
        self.target_depth = max_len
        self.candidate_lane_edge_ids = route_ids
        self.max_path_len = max_len

    def get_candidate_paths(self, edges, observation, map_infos=None):
        '''Get candidate paths using depth first search'''
        # get all paths
        paths = []
        for edge in edges:
            paths.extend(self.depth_first_search(edge,observation,map_infos=map_infos))

        candidate_paths = {}

        # extract path polyline
        for i, path in enumerate(paths):
            path_polyline = []
            for edge in path:
                poly = observation['map_polylines'][0,edge,:,:]
                poly_mask = observation['map_polylines_mask'][0,edge,:]
                path_polyline.extend(poly[poly_mask,:])
                
            path_polyline = np.array(path_polyline)
            path_polyline = self.check_path(np.array(LineString(path_polyline[:,:2]).coords))
            dist_to_ego = scipy.spatial.distance.cdist([self.ego_point], path_polyline[:, :2])
            path_polyline = path_polyline[dist_to_ego.argmin():]
            if len(path_polyline) < 3:
                continue

            path_len = len(path_polyline) * 0.25
            polyline_heading =self.calculate_path_heading(path_polyline)
            path_polyline = np.stack([path_polyline[:, 0], path_polyline[:, 1], polyline_heading], axis=1)
            candidate_paths[i] = (path_len, dist_to_ego.min(), path, path_polyline)

        if len(candidate_paths) == 0:
            return None

        # trim paths by length
        self.path_len = max([v[0] for v in candidate_paths.values()])
        acceptable_path_len = MAX_LEN * 0.2 if self.path_len > MAX_LEN * 0.2 else self.path_len
        candidate_paths = {k: v for k, v in candidate_paths.items() if v[0] >= acceptable_path_len}

        # sort paths by distance to ego
        candidate_paths = sorted(candidate_paths.items(), key=lambda x: x[1][1])

        return candidate_paths

    def get_candidate_edges(self, starting_block, ego_state, map_infos=None):
        '''Get candidate edges from the starting block'''
        edges = []
        edges_distance = []
        self.ego_point = (ego_state[20,0], ego_state[20,1])
        self.num_edges = len(starting_block)

        for edge in starting_block:
            

            
            polyline =map_infos['map_polylines'][0,edge,:,:]
            dist_to_ego = scipy.spatial.distance.cdist([self.ego_point], polyline[:, :2])
            min_dist = dist_to_ego.min()
            edges_distance.append(min_dist)
            if min_dist < 8:
                edges.append(edge)
        
        # if no edge is close to ego, use the closest edge
        if len(edges) == 0:
            if starting_block:
                edges.append(starting_block[np.argmin(edges_distance)])

        return edges

    def plan(self, ego_state, starting_block, observation, traffic_light_data, map_infos):
        # Get candidate paths
        
        edges = self.get_candidate_edges(starting_block, ego_state, observation)
        candidate_paths = self.get_candidate_paths(edges, observation,map_infos)

        if candidate_paths is None:
            return None

        # Get obstacles

        tracks = observation.get('obj_trajs')
      

        obstacles = []
        vehicles = []
        for obj_traj in tracks[0,:,:,:]:
            if self.get_rectanle_polygon(obj_traj).distance(self.get_rectanle_polygon(ego_state)) > 30:
                continue

            if obj_traj[20,6] == 1:
                if np.linalg.norm(np.array((obj_traj[20,35], obj_traj[20,36]))) < 0.01:
                    obstacles.append(obj_traj)
                else:
                    vehicles.append(obj_traj)
            else:
                obstacles.append(obj_traj)

        # Generate paths using state lattice
        paths = self.generate_paths(ego_state, candidate_paths)

        # disable lane change in large intersections
        if len(traffic_light_data) > 0:
            self._just_stay_current = True
        elif self.num_edges >= 4 and ego_state[20,7] <= 3:
            self._just_stay_current = True
        else:
            self._just_stay_current = False

        # Calculate costs and choose the optimal path
        optimal_path = None
        min_cost = np.inf
        
        for path in paths:
            cost = self.calculate_cost(path, obstacles, vehicles)
            if cost < min_cost:
                min_cost = cost
                optimal_path = path[0]

        # Post-process the path
        ref_path = self.post_process(optimal_path, ego_state)

        return ref_path

    def generate_paths(self, ego_state, paths):
        '''Generate paths from state lattice'''
        new_paths = []
        ego_state = ego_state[20,0], ego_state[20,1], np.arctan2(ego_state[20,33], ego_state[20,34])
        
        for _, (path_len, dist, path, path_polyline) in paths:
            if len(path_polyline) > 81:
                sampled_index = np.array([1, 2, 3, 4]) 
            elif len(path_polyline) > 61:
                sampled_index = np.array([1, 2, 3])
            elif len(path_polyline) > 41:
                sampled_index = np.array([1, 2]) 
            elif len(path_polyline) > 21:
                sampled_index = [1]
            else:
                sampled_index = [1]
     
            target_states = path_polyline[sampled_index].tolist()
            for j, state in enumerate(target_states):
                first_stage_path = calc_4points_bezier_path(ego_state[0], ego_state[1], ego_state[2], 
                                                            state[0], state[1], state[2], 3, sampled_index[j])[0]
                second_stage_path = path_polyline[sampled_index[j]+1:, :2]
                path_polyline = np.concatenate([first_stage_path, second_stage_path], axis=0)
                new_paths.append((path_polyline, dist, path, path_len))     

        return new_paths

    def calculate_cost(self, path, obstacles, vehicles):
        # path curvature
        curvature = self.calculate_path_curvature(path[0][:100])
        curvature = np.max(curvature)

        # lane change
        lane_change = path[1]
        if self._just_stay_current:
            lane_change = 5 * lane_change

        # go to the target lane
        target = self.check_target_lane(path[0][:50], path[3], vehicles)

        # check obstacles
        obstacles = self.check_obstacles(path[0][:100], obstacles)

        # out of boundary
        #out_boundary = self.check_out_boundary(path[0][:100], path[2])
        
        # final cos
        #cost = (10 * obstacles) + 2 * out_boundary + 1 * lane_change  + 0.1 * curvature - 5 * target
        cost =   1 * lane_change + 10 * obstacles + 0.1 * curvature - 5 * target

        return cost

    def post_process(self, path, ego_state):
        index = np.arange(0, len(path), 10)
        x = path[:, 0][index]
        y = path[:, 1][index]
        rx, ry, ryaw, rk = calc_spline_course(x, y)
        spline_path = np.stack([rx, ry, ryaw, rk], axis=1)
        
        ref_path = spline_path[:self.max_path_len*10]

        return ref_path

    def depth_first_search(self, starting_edge,observation, depth=0, map_infos=None):
        from datasets.base_dataset import BaseDataset
        if depth >= self.target_depth:
            return [[starting_edge]]
        else:
            traversed_edges = []
            
            child_edges = []
            succ = BaseDataset.get_succesor_edges_bis(observation.get('map_polylines'),observation.get('map_polylines_mask'), [starting_edge]).get(str(starting_edge)).get('successor')
            if succ:
                for edge in succ:
                    prox = self.candidate_lane_edge_ids                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                    if edge in prox:
                        child_edges.append(edge)

            if child_edges:
                for child in child_edges:
                        
                            
                        edge_len = self.get_polyline_length(observation['map_polylines'][0,child,:,:2])
                        traversed_edges.extend(self.depth_first_search(child,observation, depth+edge_len, map_infos))
            if len(traversed_edges) == 0:
                return [[starting_edge]]

            edges_to_return = []

            for edge_seq in traversed_edges:
                edges_to_return.append([starting_edge] + edge_seq)
                    
            return edges_to_return
        

    def get_polyline_length(self, polyline):
        '''Get the length of the polyline'''
        length = 0
        for i in range(1, polyline.shape[0]):
            length += np.linalg.norm(polyline[i,:2] - polyline[i-1,:2])

        return length
        
    def check_target_lane(self, path, path_len, vehicles):
        if np.abs(path_len - self.path_len) > 5:
            return 0
        
        expanded_path = LineString(path).buffer((4/2), cap_style=CAP_STYLE.square)
        min_distance_to_vehicles = np.inf

        for v in vehicles:
            v_polygon = self.get_rectanle_polygon(v)
            d = expanded_path.distance(v_polygon)
            if d < min_distance_to_vehicles:
                min_distance_to_vehicles = d

        if min_distance_to_vehicles < 5:
            return 0

        return 1

    @staticmethod
    def check_path(path):
        refine_path = [path[0]]
        
        for i in range(1, path.shape[0]):
            if np.linalg.norm(path[i] - path[i-1]) < 0.1:
                continue
            else:
                refine_path.append(path[i])
        
        line = np.array(refine_path)

        return line

    @staticmethod
    def calculate_path_heading(path):
        heading = np.arctan2(path[1:, 1] - path[:-1, 1], path[1:, 0] - path[:-1, 0])
        heading = np.append(heading, heading[-1])

        return heading
    
    @staticmethod
    def check_obstacles(path, obstacles):
        expanded_path = LineString(path).buffer((4/2), cap_style=CAP_STYLE.square)

        for obstacle in obstacles:
            obstacle_polygon = LatticePlanner.get_rectanle_polygon(obstacle)
            if expanded_path.intersects(obstacle_polygon):
                return 1

        return 0
    
    @staticmethod
    def check_out_boundary(polyline, path,observation=None):
        line = LineString(polyline).buffer((4/2), cap_style=CAP_STYLE.square)

        map= observation['map_polylines'][0]
        masks= observation['map_polylines_mask'][0]

        for road, mask in zip(map, masks):
            road = road[mask]
            if road[0, 9]==1:
                road = LineString(road[:,:2]).buffer((4/2), cap_style=CAP_STYLE.square)
                if line.intersects(road) :
                    return 1

        return 0

    @staticmethod
    def calculate_path_curvature(path):
        dx = np.gradient(path[:, 0])
        dy = np.gradient(path[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        curvature = np.abs(dx * d2y - d2x * dy) / (dx**2 + dy**2)**(3/2)

        return curvature

    
        
    @staticmethod
    def get_rectanle_polygon(ego_state):
        '''Get the rectangle polygon of the ego vehicle'''
        length, width = ego_state[20,3], ego_state[20,4]
        ego_x, ego_y, ego_h = ego_state[20,0], ego_state[20,1], ego_state[20,6]
        ego_polygon = np.array([[-length/2, -width/2], [-length/2, width/2], [length/2, width/2], [length/2, -width/2]])
        rotation_matrix = np.array([[np.cos(ego_h), -np.sin(ego_h)], [np.sin(ego_h), np.cos(ego_h)]])
        ego_polygon = np.dot(ego_polygon, rotation_matrix.T)
        ego_polygon += np.array([ego_x, ego_y])

        ego_polygon = Polygon(ego_polygon)

        return ego_polygon