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
    [0.1, 0.2, 0]
])

# Instantiate Robotarium object
r = robotarium.Robotarium(number_of_robots=N, sim_in_real_time=True, initial_conditions=initial_conditions.T)

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
w_h = 1  # Weight for quality (health) cost
w_r = 1  # Weight for capacity cost
# Robot position tracking
previous_centroids = np.zeros((N, 2))
weights = np.ones(N)  # Initialize uniform weights
costs = np.zeros(N)
alpha = 0.1  # Adaptive weight update rate

# Grid setup for Voronoi
x_min, x_max, y_min, y_max, res = -1.5, 1.5, -1, 1, 0.05
# Function to compute the cost for each robot
N1 = list(N1)  # Convert N1 to a list if it's a set
N2 = list(N2)  # Convert N2 to a list if it's a set
def compute_cost(i):
    # Velocity cost
    velocity_cost = Vr[i - 1] * w_v  # Velocity for robot i
    
    # Quality and Capacity cost calculation based on the robot type
    if i in N1:  # If robot i belongs to sensor type 1
        quality_cost = hi1[N1.index(i)] * w_h  # Quality cost for N1 robots
        capacity_cost = Rrs1[N1.index(i)] * w_r  # Capacity cost for N1 robots
    else:  # If robot i belongs to sensor type 2
        quality_cost = hi2[N2.index(i)] * w_h  # Quality cost for N2 robots
        capacity_cost = Rrs2[N2.index(i)] * w_r  # Capacity cost for N2 robots
    
    # Average over the sensor types (since |S| = 2)
    sensor_type_cost = (quality_cost + capacity_cost) / len(S)

    # Total cost: c_i = w_v(v_i) + (1/|S|) * sum(t_si * [w_h(h_si) + w_r(r_si)])
    total_cost = velocity_cost + sensor_type_cost
    return total_cost

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
            min_index = np.argmin(distances)
            c_v[min_index] += [ix, iy]
            w_v[min_index] += 1

    si_velocities = np.zeros((2, N))
    new_centroids = np.copy(previous_centroids)

    for robot in range(N):
        if w_v[robot] != 0:
            new_centroids[robot] = c_v[robot] / w_v[robot]
            si_velocities[:, robot] = k_p * (new_centroids[robot] - [current_x[robot, 0], current_y[robot, 0]])

    for i in range(1,N):  # For robots 1 to 5
        cost = compute_cost(i)
        print(f"Cost for robot {i}: {cost}")

    # Adaptive Weight Update
    for i in range(N):
        weight_update = alpha * sum((weights[j] - costs[j]) for j in range(N) if j != i)
        weights[i] += weight_update
        weights[i] = max(0.1, weights[i])  # Prevent negative weights

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
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.scatter(current_x, current_y, c='red', label='Robots')
    ax.set_title(f'Weighted Voronoi - Iteration {k+1}')
    ax.legend()
    plt.pause(0.1)

plt.ioff()
plt.show()

r.call_at_scripts_end()
