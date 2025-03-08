import sys
sys.path.append('..')

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

convergence_threshold = 2e-3         # Threshold to determine if convergence has occurred
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
kv = 1               # Constant Gain for velocity controller
kw = 1               # Constant Gain for weights controller (Only used for Power Cost Calculation)
kc = 1               # Constant Gain for our algorithms cost weighted controller
w_v = 1              # Weight for velocity cost
w_h = 1              # Weight for sensor quality cost
w_r = 1              # Weight for sensor capacity cost
alpha = 0.1          # Adaptive weight update rate
curve_factor = 0.01  # Factor to add to the curvature to the movement
S = {1}              # Set of sensor types
N1 = {1,2,3,4,5}     # Set of robots with sensor type 1
N2 = {}              # Set of robots with sensor type 1
hi1 = [1,1,1,1,1]    # Set sensor health for sensor type 1
hi2 = []             # Set sensor health for sensor type 2
v_r = [1,1,1,1,1]    # Velocities set to 1 initially for all robots
Rrs1 = [1,1,1,1,1]   # Set sensor health for sensor type 1
Rrs2 = []            # Set sensor health for sensor type 2
hi = hi1 + hi2       # Calculate overall sensor health array
Rrs = Rrs1 + Rrs2    # Calculate overall sensor capacity array
wi = np.zeros(N)     # Weights initially set to zero and calculated later (Only used for Power Cost Calculation)
weights= np.zeros(N) # Cost Weights initially set to zero and calculated later
costs = np.zeros(N)  # Costs initially set to zero and calculated based on values above
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
total_Hc = []
previous_x = None
previous_y = None
converged_iteration = -1

# Instantiate Robotarium object
r = robotarium.Robotarium(number_of_robots=N, sim_in_real_time=True, initial_conditions=initial_conditions[0:N].T)

# Helper Functions for Simulation
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# Visualization setup
fig, ax = plt.subplots()

def compute_cost(i):
    """
    Function to compute the cost of the robot based on the predefined sensor qualities,
    velocities, sensor capacities, and their corresponding weights
    """
    # Calculate the velocity cost
    velocity_cost = v_r[i - 1] * w_v

    # Calculate the sensor quality and capacity cost for each sensor type
    if i in N1:
        quality_cost_1 = hi1[list(N1).index(i)] * w_h
        capacity_cost_1 = Rrs1[list(N1).index(i)] * w_r
        sensor_cost_1 = quality_cost_1 + capacity_cost_1
    else:
        sensor_cost_1 = 0
    if i in N2:
        quality_cost_2 = hi2[list(N2).index(i)] * w_h
        capacity_cost_2 = Rrs2[list(N2).index(i)] * w_r
        sensor_cost_2 = quality_cost_2 + capacity_cost_2
    else:
        sensor_cost_2 = 0

    # Add both sensor costs for each type, necessary if a robot has both sensors
    sensor_cost = (sensor_cost_1/len(S)) + (sensor_cost_2/len(S))

    # Calculate the total cost
    total_cost = velocity_cost + sensor_cost

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
    for robot in range(N):

        # Instantiate the summation variable to zero for the next calculation
        summation = 0

        # Calculate the summation for the weights by comparing the cost of the current robot to all of the others
        for neighbor in range(N):
            summation += (costs[robot] - weights[robot]) - (costs[neighbor] - weights[neighbor])
        
        # Calculate the weights
        weights[robot] = (kc/(2*w_vw[robot])) * summation

        # # Calculate the weight diff
        # weight_diff = summation
        
        # # Update the weight based on the cost difference
        # weight_update = alpha * weight_diff
        
        # # Adjust weight dynamically during decision-making
        # # If cost is high, prioritize improving that robot's performance
        # if costs[i] > 1.0:  # Example: If cost is high, decrease weight to prioritize improvement
        #     weights[i] -= weight_update * 0.5  # Penalize robots with high cost slightly
        # else:
        #     weights[i] += weight_update * 0.5  # Reward robots with lower cost
        
        # # Apply damping factor to avoid drastic changes in weights
        # weights[i] += weight_update * damping_factor
        
        # # Ensure weights stay within a reasonable range
        # weights[i] = max(weight_limit[0], min(weight_limit[1], weights[i]))  # Enforce min and max limits
        
        # # Penalize overperforming robots (adaptive nature)
        # if weights[i] > 5:  # Example threshold for re-adjustment, can be changed
        #     weights[i] -= 0.1  # Slight reduction for robots performing too well, to balance task distribution
    
    print("Weights:", weights)

    return weights

# Compute cost  for each robot
for i in range(1, N + 1):
    costs[i - 1] = compute_cost(i)

print(f"Computed Costs: {costs}")

# Iteration loop
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
    Hc = 0
    c_vc = np.zeros((N, 2))
    w_vc = np.zeros(N)
    w_vw = np.zeros(N)
    cwi = np.zeros((2,N))
    si_velocities = np.zeros((2,N))
    new_centroids = np.zeros((N,2))
    distance_traveled = np.zeros(N)

    # Calculate each robot's sensor range
    ranges = np.array(Rrs) * max_range

    # Iterate through all of the points in the domain, this is the calculation part of the code for the centroids, masses, and costs
    for ix in np.arange(x_min, x_max, res):
        for iy in np.arange(y_min, y_max, res):

            # Set the importance value to 1, this represents the distribution function
            importance_value = 1

            # Calculate the distances
            distances = np.linalg.norm(np.column_stack((current_x, current_y)) - np.array([ix, iy]), axis=1)
            
            # Subtract the weights from the normalized distances to find the weighted Voronoi Partitions
            weighted_distances = distances.copy()
            weighted_distances -= wi

            # Divide the velocities from the normalized distances to find the temporal Voronoi Partitions
            velocity_distances = distances.copy()
            velocity_distances = velocity_distances/v_r

            # Subtract the costs from the normalized distances to find our custom Voronoi Partitions
            cost_weighted_distances = distances.copy()
            cost_weighted_distances -= weights

            # Find the minimum indexes for the different Voronoi partitions
            min_index = np.argmin(distances)
            weighted_min_index = np.argmin(weighted_distances)
            velocity_min_index = np.argmin(velocity_distances)
            cost_weighted_min_index = np.argmin(cost_weighted_distances)

            # Calculate the centroids and masses for our custom algorithm
            c_vc[cost_weighted_min_index] += [importance_value * ix, importance_value * iy]
            w_vc[cost_weighted_min_index] += importance_value

            # Calculate sensor capacity and mass of weighted Voronoi partitions for cost calculations below
            sensor_capacity = ranges[min_index]/2
            w_vw[weighted_min_index] += importance_value

            # Calculate other Cost Values
            for sensor_type in S:
                if sensor_type == 1:
                    type_distances = []
                    for robot in N1:
                        current_distance = np.sqrt(np.square(ix - current_x[robot-1]) + np.square(iy - current_y[robot-1]))
                        type_distances.append(current_distance)
                    min_index = np.argmin(type_distances)
                    Hg += (type_distances[min_index] ** 2) * importance_value
                elif sensor_type == 2:
                    type_distances = []
                    for robot in N2:
                        current_distance = np.sqrt(np.square(ix - current_x[robot-1]) + np.square(iy - current_y[robot-1]))
                        type_distances.append(current_distance)
                    min_index = np.argmin(type_distances)
                    Hg += (type_distances[min_index] ** 2) * importance_value
            Hp += 0.5 * ((distances[weighted_min_index] ** 2) - wi[weighted_min_index]) * importance_value
            Hv += ((distances[velocity_min_index]/v_r[velocity_min_index]) ** 2) * importance_value
            if distances[min_index] > sensor_capacity:
                Hr += (sensor_capacity ** 2) * importance_value
            if distances[min_index] <= sensor_capacity:
                Hr += (distances[min_index] ** 2) * importance_value
            Hc += 0.5 * ((distances[cost_weighted_min_index] ** 2) - weights[cost_weighted_min_index]) * importance_value

    # Make a copy of the current weights (Only used for Power Cost Calculation)
    wi_copy = wi.copy()

    # Iterate for the number of robots, this is the velocity and weight controller portion of the code
    for robot in range(N):

        # Instantiate the summation variable to zero for the next calculations (Only used for Power Cost Calculation)
        summation = 0

        # Calculate the summation for the weights (Only used for Power Cost Calculation)
        for neighbor in range(N):
            summation += (hi[robot] - wi_copy[robot]) - (hi[neighbor] - wi_copy[neighbor])
        
        # Calculate the new weights (Only used for Power Cost Calculation)
        wi[robot] += (kw/(2*w_vw[robot])) * summation

        # Calculate the distance travled from the previous iteration
        if previous_x is not None and previous_y is not None:
            distance_traveled[robot] = float(np.sqrt((abs(current_x[robot][0] - previous_x[robot][0]) ** 2) + \
                                                      (abs(current_y[robot][0] - previous_y[robot][0]) ** 2)))

        if w_vc[robot] != 0:

            # Calculate the new centroids
            new_centroids[robot] = c_vc[robot] / w_vc[robot]

            # Apply curvature by modifying the velocity direction slightly
            direction_to_target = np.array([new_centroids[robot][0] - current_x[robot, 0], 
                                           new_centroids[robot][1] - current_y[robot, 0]])
            norm_direction = np.linalg.norm(direction_to_target)
            curvature_adjustment = curve_factor * np.array([-direction_to_target[1], direction_to_target[0]])  # Perpendicular vector for curved motion
            
            # Calcualte the velocities of each robot
            si_velocities[:, robot] = kv * (direction_to_target + curvature_adjustment / norm_direction)
            # si_velocities[:, robot] = kv * (direction_to_target)

            # Append the current centroid and velocity to the lists
            cwi[:, robot] = np.array(new_centroids[robot])

    # Call adaptive weight update
    weights = adaptive_weight_update(N, weights, costs)

    # Make a copy of distance traveled array to calculate total distances
    total_distances = distance_traveled.copy()

    # If there is a value in the dist array, add the latest calculated distance to the previous iteration
    if dist_robot:
        total_distances += dist_robot[-1]

    # Update the variables for the previous pose with the current pose to be used in the next calculation
    previous_x = current_x
    previous_y = current_y

    # Add the current iteration values to the global lists 
    wij.append(weights.tolist())
    Cwi.append(cwi.tolist())
    ui.append(si_velocities.tolist())
    dist_robot.append(total_distances.tolist())
    total_Hg.append(Hg[0])
    total_Hp.append(Hp)
    total_Hv.append(Hv)
    total_Hr.append(Hr)
    total_Hc.append(Hc)

    # Check for convergence
    if np.all(distance_traveled < convergence_threshold) and k > 3:
        converged_iteration = k + 1
        print(f"Converged at iteration {converged_iteration}")
        break

    # Transform and assign the velocities, then iterate the simulation
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

# Outputs the data to a .csv file
with open("output5.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([f"Number of Iterations: {converged_iteration}"])
    writer.writerow(["X Poses", "Y Poses"])
    for index, value in enumerate(poses, start=1):
        writer.writerow([f"Iteration {index}", value])
    writer.writerow([])
    writer.writerow(["Weights"])
    for index, value in enumerate(wij, start=1):
        writer.writerow([f"Iteration {index}", value])
    writer.writerow([])
    writer.writerow(["Centroids"])
    for index, value in enumerate(Cwi, start=1):
        writer.writerow([f"Iteration {index}", value])
    writer.writerow([])
    writer.writerow(["Control Inputs"])
    for index, value in enumerate(ui, start=1):
        writer.writerow([f"Iteration {index}", value])
    writer.writerow([])
    writer.writerow(["Distance Traveled"])
    for index, value in enumerate(dist_robot, start=1):
        writer.writerow([f"Iteration {index}", value])
    writer.writerow([])
    writer.writerow(["Locational Cost for all Sensor Types"])
    for index, value in enumerate(total_Hg, start=1):
        writer.writerow([f"Iteration {index}: {value}"])
    writer.writerow([])
    writer.writerow(["Power Cost"])
    for index, value in enumerate(total_Hp, start=1):
        writer.writerow([f"Iteration {index}: {value}"])
    writer.writerow([])
    writer.writerow(["Temporal Cost"])
    for index, value in enumerate(total_Hv, start=1):
        writer.writerow([f"Iteration {index}: {value}"])
    writer.writerow([])
    writer.writerow(["Range-Limited Cost"])
    for index, value in enumerate(total_Hr, start=1):
        writer.writerow([f"Iteration {index}: {value}"])
    writer.writerow([])
    writer.writerow(["Custom Heterogeneous Cost"])
    for index, value in enumerate(total_Hc, start=1):
        writer.writerow([f"Iteration {index}: {value}"])
    writer.writerow([])
    writer.writerow(["Proposed Algorithm"])

r.call_at_scripts_end()