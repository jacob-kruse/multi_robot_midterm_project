import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

import rps.robotarium as robotarium
from rps.utilities.graph import *
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

# Simulation Variables
N = 5                    # Number of robots
iterations = 1000        # How many iterations do we want (about N*0.033 seconds)
L = lineGL(N)            # Generated a connected graph Laplacian (for a cylce graph).
x_min = -1.5             # Upper bound of x coordinate
x_max = 1.5              # Lower bound of x coordinate
y_min = -1               # Upper bound of y coordinate
y_max = 1                # Lower bound of x coordinate
res = 0.05               # Resolution of coordinates

convergence_threshold = 1e-2         # Threshold to determine if convergence has occurred
initial_conditions = np.asarray([    # Sets the initial positions of the robots
    [1.25, 0.25, 0],
    [1, 0.5, 0],
    [1, -0.5, 0],
    [-1, -0.75, 0],
    [0.1, 0.2, 0],
    [0.2, -0.6, 0],
    [-0.75, -0.1, 0],
    [-1, 0, 0],
    [-0.8, -0.25, 0],
    [1.3, -0.4, 0]
])

# Scenario Variables
k = 1               # Constant Gain for velocity controller
kw = 1              # Constant Gain for weights controller
ns = 2              # Number of sensor types
S = {1}             # Set of sensor types
N1 = {1,2,3,4,5}    # Set of robots with sensor type 1
N2 = {}             # Set of robots with sensor type 1
hi = np.ones(N)     # Sensor quality set to 1 initially for all sensors
# hi[2] = 0.5         # Sensor quality of 3rd robot is 0.5
wi = np.zeros(N)    # Weights initially set to zero and calculated later
v_r = np.ones(N)    # Velocities set to 1 initially for all robots
Rrs1 = np.ones(N)   # Sensor Capacity, set to 1 initially for all sensors
# Rrs1[2] = 0.5       # Set the Sensor Capacity of Robot 3 to 0.5
max_range = np.sqrt(((x_max-x_min) ** 2) + ((y_max-y_min) ** 2)) # Calculate the maximum range to cover entire simulation

# Status and Performance Variables
poses = []
wij = []
Cwi = []
ui = []
dist_robot = []
total_Hg = []
total_Hp = []
total_Hv = []
total_Hr = []
previous_x = None
previous_y = None
previous_H_v = float('inf')
converged_iteration = -1
previous_centroids = np.zeros((N, 2))

# Instantiate Robotarium object
r = robotarium.Robotarium(number_of_robots=N, sim_in_real_time=True, initial_conditions=initial_conditions[0:N].T)

def get_sensor():
    """
    Function to return the sensor density measurement at a given location q.
    Currently returns a constant value of 1 (uniform distribution).
    """
    return 1

si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# Visualization setup
# plt.ion()
fig, ax = plt.subplots()

# Helper function to compute multiplicatively weighted Voronoi partitioning
def weighted_voronoi_partition(positions, velocities, grid_points):
    """
    Compute the multiplicatively weighted Voronoi partitioning for each point.
    Each robot is assigned a region based on its weighted distance from grid points.
    """
    regions = np.zeros(grid_points.shape[0], dtype=int)  # List to store region assignments
    for idx, point in enumerate(grid_points):
        min_distance = np.inf
        assigned_robot = -1
        
        for i in range(N):
            # Compute the distance between the point and robot i
            distance = np.linalg.norm(point - positions[i])
            weighted_distance = distance / velocities[i]  # Weight the distance by velocity
            
            if weighted_distance < min_distance:
                min_distance = weighted_distance
                assigned_robot = i
        
        regions[idx] = assigned_robot  # Assign the point to the closest robot's region
    
    return regions

# Compute Centroid of the Multiplicatively Weighted Voronoi Regions
def compute_centroids(p, velocities, grid_points):
    regions = weighted_voronoi_partition(p, velocities, grid_points)
    centroids = np.zeros((N, 2))  # Array to store centroids
    
    # Compute the centroid for each robot based on the weighted Voronoi partition
    for robot in range(N):
        # Get all points assigned to robot's region
        region_points = grid_points[regions == robot]
        if region_points.shape[0] > 0:
            centroids[robot] = np.mean(region_points, axis=0)  # Compute centroid as the mean of the region points
            
    return centroids

# TEMPORAL COST (Locational cost for each robot based on the weighted Voronoi partition)
def temporal_cost(p, velocities, grid_points, phi_func):
    total_cost = 0
    regions = weighted_voronoi_partition(p, velocities, grid_points)
    
    # Iterate over each robot's region and compute the temporal cost
    for i in range(N):
        region_points = grid_points[regions == i]
        if region_points.shape[0] > 0:
            for q in region_points:
                distance = np.linalg.norm(q - p[i])
                velocity_weight = 1 / velocities[i]**2  # Weighting by velocity squared
                phi_q = phi_func(q)  # Apply the weighting function φ(q)
                total_cost += (distance**2) * velocity_weight * phi_q
    
    return total_cost

# Define the weighting function φ(q) (Example: Gaussian function)
def phi(q):
    return 1

# Compute the position update and control law for each robot
def position_update(p, centroids, velocities, k_i=1):
    # Initialize velocity array
    velocities_update = np.zeros_like(p)
    
    for i in range(N):
        # Compute the distance between the robot's position and its centroid
        distance = np.linalg.norm(p[i] - centroids[i])
        
        # Compute the control gain for robot i
        k_hat = min(velocities[i] / (k_i * distance), 1) * k_i
        
        # Compute the velocity update based on the control law
        velocities_update[i] = -k_hat * (p[i] - centroids[i])
    
    return velocities_update

# Main loop for controlling the robots
for k in range(iterations):
    x = r.get_poses()
    x_si = uni_to_si_states(x)
    poses.append(x_si.tolist())
    current_x = x_si[0, :, None]
    current_y = x_si[1, :, None]
    
    # Instantiate calculation variables to zero
    Hg = 0
    Hp = 0
    Hv = 0
    Hr = 0
    c_vw = np.zeros((N,2))
    w_vw = np.zeros(N)
    cwi = np.zeros((2,N))
    distance_traveled = np.zeros(N)
    
    # Update the positions of the robots based on their control laws
    centroids = compute_centroids(np.column_stack((current_x, current_y)), v_r, np.array([[ix, iy] for ix in np.arange(x_min, x_max, res) for iy in np.arange(y_min, y_max, res)]))
    
    # Apply position update based on the control law
    velocities_update = position_update(np.column_stack((current_x, current_y)), centroids, v_r)
    
    # Update velocities and move the robots
    dxu = si_to_uni_dyn(velocities_update.T, x)
    r.set_velocities(np.arange(N), dxu)
    r.step()

    # Compute the temporal cost (locational cost)
    Hv = temporal_cost(np.column_stack((current_x, current_y)), v_r, np.array([[ix, iy] for ix in np.arange(x_min, x_max, res) for iy in np.arange(y_min, y_max, res)]), phi)
    print(f"Iteration {k + 1}: H_v = {Hv}")
    if abs(previous_H_v - Hv) < convergence_threshold:
        converged_iteration = k + 1
        print(f"Converged at iteration {converged_iteration} with H_v = {Hv}")
        break
    
    previous_H_v = Hv
    # Visualization (same as before)
    
    ax.clear()
    points = np.column_stack((current_x[:, 0], current_y[:, 0]))
    if len(points) > 2:
        vor = Voronoi(points)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=1.5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.scatter(current_x, current_y, c='red', label='Robots')
    ax.set_title(f'Voronoi Partitioning with Weighted Controller - Iteration {k+1} - Temporal Cost: {Hv:.4f}')
    ax.legend()
    plt.pause(0.1)

plt.ioff()
plt.show()

r.call_at_scripts_end()