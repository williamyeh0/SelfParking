import numpy as np
import heapq
from math import sin, cos, tan, pi

""" 
should base this off of the gnss_tracker_pp since it uses waypoints, PPC
"""

class Node:
    def __init__(self, x, y, yaw, direction, steer, cost, parent_index):
        self.x = x              # x position
        self.y = y              # y position 
        self.yaw = yaw          # yaw angle
        self.direction = direction  # driving direction (forward=1, backward=-1)
        self.steer = steer      # steering angle
        self.cost = cost        # cost to reach this node
        self.parent_index = parent_index
        
class HybridAStar:
    def __init__(self):
        # Vehicle parameters
        self.wheelbase = 1.75     # GEM wheelbase in meters
        self.max_steer = 0.61     # Maximum steering angle in radians (~35 degrees)
        self.min_turn_radius = self.wheelbase/tan(self.max_steer)
        
        # Search parameters
        self.resolution = 0.5      # Grid resolution
        self.yaw_resolution = np.deg2rad(15.0)  # Yaw angle resolution
        self.motion_resolution = 0.1  # Path interpolation resolution
        self.n_steer = 20         # Number of steer angles to check
        
        # Cost weights
        self.steer_cost = 1.0     # Steering penalty
        self.steer_change_cost = 5.0  # Steering change penalty
        self.backward_cost = 5.0   # backward penalty
        self.direction_change_cost = 10.0  # Direction change penalty
        
    def plan_path(self, start_x, start_y, start_yaw, 
                  goal_x, goal_y, goal_yaw, 
                  ox, oy):  # obstacle x,y coordinates
        """
        Main path planning function using Hybrid A*
        Returns path if found, None otherwise
        """
        start_node = Node(start_x, start_y, start_yaw, True, 0.0, 0.0, None)
        goal_node = Node(goal_x, goal_y, goal_yaw, True, 0.0, 0.0, None)
        
        # Initialize open/closed sets
        open_set = {}
        closed_set = {}
        pq = []  # Priority queue
        
        # Add start node
        start_id = self.calc_index(start_node)
        open_set[start_id] = start_node
        heapq.heappush(pq, (0, start_id))
        
        while True:
            if not open_set:
                print("Cannot find path - Open set is empty")
                return None
                
            # Get node with minimum cost
            cost, current_id = heapq.heappop(pq)
            current = open_set[current_id]
            
            # Remove from open set and add to closed set
            del open_set[current_id]
            closed_set[current_id] = current
            
            # Check if we reached goal
            if self.is_goal(current, goal_node):
                goal_node.parent_index = current_id
                goal_node.cost = current.cost
                print("Path found!")
                return self.extract_path(closed_set, goal_node)
            
            # Expand search grid based on vehicle motion
            for next_node in self.get_next_nodes(current, ox, oy):
                node_ind = self.calc_index(next_node)
                
                # If already in closed set, skip
                if node_ind in closed_set:
                    continue
                    
                if node_ind not in open_set:
                    open_set[node_ind] = next_node
                    heapq.heappush(pq, (next_node.cost, node_ind))
                else:
                    if open_set[node_ind].cost > next_node.cost:
                        open_set[node_ind] = next_node
                        heapq.heappush(pq, (next_node.cost, node_ind))
                        
    def get_next_nodes(self, current, ox, oy):
        """Generate next nodes considering vehicle motion"""
        nodes = []
        
        # Discrete steer angles
        for steer in np.linspace(-self.max_steer, self.max_steer, self.n_steer):
            # Forward and backward movement
            for direction in [1, -1]:  # forward and backward
                node = self.calc_next_node(current, steer, direction, ox, oy)
                if node:
                    nodes.append(node)
                    
        return nodes
        
    def calc_next_node(self, current, steer, direction, ox, oy):
        """Calculate next node based on bicycle model"""
        x, y, yaw = current.x, current.y, current.yaw
        
        # Update state using bicycle model
        distance = self.motion_resolution * direction
        x += distance * cos(yaw)
        y += distance * sin(yaw)
        yaw += distance * tan(steer) / self.wheelbase
        
        # Normalize yaw angle
        yaw = self.normalize_angle(yaw)
        
        # Check collision
        if self.check_collision(x, y, ox, oy):
            return None
            
        # Calculate cost
        cost = current.cost + self.motion_resolution
        cost += self.steer_cost * abs(steer)
        cost += self.steer_change_cost * abs(current.steer - steer)
        if direction != current.direction:
            cost += self.direction_change_cost
        if direction == -1:
            cost += self.backward_cost
            
        node = Node(x, y, yaw, direction, steer, cost, None)
        node.parent_index = self.calc_index(current)
        
        return node
        
    def normalize_angle(self, angle):
        """Normalize angle between -pi and pi"""
        while angle > pi:
            angle -= 2.0 * pi
        while angle < -pi:
            angle += 2.0 * pi
        return angle
        
    def calc_index(self, node):
        """Calculate unique index for each node based on position and yaw"""
        x_ind = round(node.x / self.resolution)
        y_ind = round(node.y / self.resolution)
        yaw_ind = round(node.yaw / self.yaw_resolution)
        return (x_ind, y_ind, yaw_ind)
        
    def is_goal(self, current, goal):
        """Check if current node is close enough to goal"""
        dx = current.x - goal.x
        dy = current.y - goal.y
        dyaw = abs(self.normalize_angle(current.yaw - goal.yaw))
        
        goal_tolerance = 0.5
        yaw_tolerance = np.deg2rad(15.0)
        
        if dx*dx + dy*dy < goal_tolerance*goal_tolerance and dyaw < yaw_tolerance:
            return True
        return False
        
    def check_collision(self, x, y, ox, oy):
        """Basic collision check - can be made more sophisticated"""
        for ix, iy in zip(ox, oy):
            dx = x - ix
            dy = y - iy
            if dx*dx + dy*dy < 1.0:  # collision threshold
                return False
        return True
        
    def extract_path(self, closed_set, goal_node):
        """Extract the path from start to goal"""
        path_x = [goal_node.x]
        path_y = [goal_node.y]
        path_yaw = [goal_node.yaw]
        
        parent_index = goal_node.parent_index
        while parent_index:
            node = closed_set[parent_index]
            path_x.append(node.x)
            path_y.append(node.y) 
            path_yaw.append(node.yaw)
            parent_index = node.parent_index
            
        return path_x[::-1], path_y[::-1], path_yaw[::-1]

if __name__ == '__main__':
    # initialize planner
    planner = HybridAStar()

    path_x, path_y, path_yaw = planner.plan_path(
        start_x, start_y, start_yaw,
        goal_x, goal_y, goal_yaw,
        obstacle_x_list, obstacle_y_list
    )