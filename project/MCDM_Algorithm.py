import rps.robotarium as robotarium
from rps.utilities.graph import *
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Number of robots (N)
N = 5
# Initial robot conditions (position and orientation)
initial_conditions = np.asarray([
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
r = robotarium.Robotarium(number_of_robots=N, sim_in_real_time=True, initial_conditions=initial_conditions[0:N].T)

# Parameters
iterations = 500
convergence_threshold = 1e-3
k_p = 1.0  # Proportional gain
# Create barrier certificates and mappings
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# Visualization setup
plt.ion()
fig, ax = plt.subplots()

# Sensor type assignments
S = {1, 2}  # Two types of sensors
N1 = {1, 2, 3}  # Sensor type 1 robots
N2 = {3, 4, 5}  # Sensor type 2 robots

# Health of robots for each sensor type (normalized)
hi1 = [1, 1, 1]
hi2 = [1, 1, 1]

# Robot velocities (m/s)
Vr = [1, 1, 1, 1, 1]

# Range of robots for each sensor type (normalized)
Rrs1 = [1, 1, 1]  # For N1 robots (Sensor type 1)
Rrs2 = [1, 1, 1]  # For N2 robots (Sensor type 2)
# Cost Weights (these could be determined based on priorities or calibration)

w_v = 1  # Weight for velocity cost
Weight_velocity=1
w_h = 1  # Weight for quality (health) cost
w_r = 1  # Weight for capacity cost
# Robot position tracking
previous_centroids = np.zeros((N, 2))
weights = np.ones(N)  # Initialize uniform weights
costs = np.zeros(N)
alpha = 0.1  # Adaptive weight update rate
curvature_factor = 0.01  # Factor to add curvature to the movement
# Grid setup for Voronoi
x_min, x_max, y_min, y_max, res = -1.5, 1.5, -1, 1, 0.05
# Function to compute the cost for each robot
N1 = list(N1)  # Convert N1 to a list if it's a set
N2 = list(N2)  # Convert N2 to a list if it's a set
def compute_cost(i):
    global w_v,w_r,w_h
    velocity_cost = Vr[i - 1] * Weight_velocity
    if i in N1:
        quality_cost = hi1[N1.index(i)] * w_h
        capacity_cost = Rrs1[N1.index(i)] * w_r
    elif i in N2:
        quality_cost = hi2[N2.index(i)] * w_h
        capacity_cost = Rrs2[N2.index(i)] * w_r
    else:
        quality_cost = 0
        capacity_cost = 0

    sensor_type_cost = (quality_cost + capacity_cost) / len(S)
    total_cost = velocity_cost + sensor_type_cost

    if np.any(velocity_cost < 1.0):
        w_h += 0.1
    elif (sensor_type_cost < 1.0):
        w_v += 0.1
    return total_cost
def adaptive_weight_update(N, weights, costs, alpha=0.1, damping_factor=0.1, weight_limit=(0.1, 10)):
    """
    Function to update the weights of robots dynamically based on cost difference and other adaptive rules.
    
    Parameters:
    - N (int): Number of robots
    - weights (list or array): Current weights of the robots
    - costs (list or array): Current costs of the robots
    - alpha (float): Learning rate or adjustment factor for weight update
    - damping_factor (float): Factor to apply a small adjustment to avoid drastic weight changes
    - weight_limit (tuple): Minimum and maximum allowed values for the weights

    Returns:
    - weights (list or array): Updated weights for all robots
    """
    for i in range(N):
        # Calculate the difference in cost for robot i from other robots
        weight_diff = np.mean(costs) - costs[i]
        
        # Update the weight based on the cost difference
        weight_update = alpha * weight_diff
        
        # Adjust weight dynamically during decision-making
        # If cost is high, prioritize improving that robot's performance
        if costs[i] > 1.0:  # Example: If cost is high, decrease weight to prioritize improvement
            weights[i] -= weight_update * 0.5  # Penalize robots with high cost slightly
        else:
            weights[i] += weight_update * 0.5  # Reward robots with lower cost
        
        # Apply damping factor to avoid drastic changes in weights
        weights[i] += weight_update * damping_factor
        
        # Ensure weights stay within a reasonable range
        weights[i] = max(weight_limit[0], min(weight_limit[1], weights[i]))  # Enforce min and max limits
        
        # Penalize overperforming robots (adaptive nature)
        if weights[i] > 5:  # Example threshold for re-adjustment, can be changed
            weights[i] -= 0.1  # Slight reduction for robots performing too well, to balance task distribution
    
    return weights

# Iteration loop
for k in range(iterations):
    x = r.get_poses()
    x_si = uni_to_si_states(x)
    current_x = x_si[0, :, None]
    current_y = x_si[1, :, None]

    c_v = np.zeros((N, 2))
    w_v = np.zeros(N)

    # Weighted Voronoi Partitioning
    for ix in np.arange(x_min, x_max, res):
        for iy in np.arange(y_min, y_max, res):
            distances = np.linalg.norm(np.column_stack((current_x, current_y)) - np.array([ix, iy]), axis=1)
            # Apply weights by subtracting from the distances
            weighted_distances = distances-weights
            min_index = np.argmin(weighted_distances)
            c_v[min_index] += [ix, iy]
            w_v[min_index] += 1

    si_velocities = np.zeros((2, N))
    new_centroids = np.copy(previous_centroids)

    for robot in range(N):
        if w_v[robot] != 0:
            new_centroids[robot] = c_v[robot] / w_v[robot]
            # Apply curvature by modifying the velocity direction slightly
            direction_to_target = np.array([new_centroids[robot][0] - current_x[robot, 0], 
                                           new_centroids[robot][1] - current_y[robot, 0]])
            norm_direction = np.linalg.norm(direction_to_target)
            curvature_adjustment = curvature_factor * np.array([-direction_to_target[1], direction_to_target[0]])  # Perpendicular vector for curved motion
            si_velocities[:, robot] = k_p * (direction_to_target + curvature_adjustment / norm_direction) * weights[robot]   # Apply dynamic weight to velocity

    # Compute costs for each robot (before task allocation)
    costs = np.zeros(N)
    for i in range(1, N + 1):
        costs[i - 1] = compute_cost(i)
    print(f"Iteration {k+1} - Computed Costs: {costs}")

    #Call adaptive weight update after costs are calculated
    weights = adaptive_weight_update(N, weights, costs)
    

    # Check for convergence
    movement = np.linalg.norm(new_centroids - previous_centroids, axis=1)
    if np.all(movement < convergence_threshold):
        print(f"Converged at iteration {k+1}")
        break
    
    previous_centroids = np.copy(new_centroids)
    dxu = si_to_uni_dyn(si_velocities, x)
    r.set_velocities(np.arange(N), dxu)
    r.step()

    # Voronoi Visualization
    ax.clear()
    points = np.column_stack((current_x[:, 0], current_y[:, 0]))
    if len(points) > 2:
        vor = Voronoi(points)
        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=1.5)
    # Add robot numbers on top of robots in the plot
    for i, point in enumerate(points):
        ax.text(point[0], point[1], f"Robot {i + 1}", fontsize=12, ha='center', color='red')

        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=1.5)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.scatter(current_x, current_y, c='red', label='Robots')
    ax.set_title(f'Weighted Voronoi - Iteration {k+1}')
    ax.legend()
    plt.pause(0.1)

plt.ioff()
plt.show()

r.call_at_scripts_end()