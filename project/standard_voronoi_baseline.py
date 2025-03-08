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
kv = 1              # Constant Gain for velocity controller
kw = 1              # Constant Gain for weights controller
kc = 1              # Constant Gain for cost controller
S = {1}             # Set of sensor types
N1 = {1,2,3,4,5}    # Set of robots with sensor type 1
N2 = {}             # Set of robots with sensor type 1
hi1 = [1,1,1,1,1]   # Set sensor health for sensor type 1
hi2 = []            # Set sensor health for sensor type 2
v_r = [1,1,1,1,1]   # Velocities set to 1 initially for all robots
Rrs1 = [1,1,1,1,1]  # Set sensor health for sensor type 1
Rrs2 = []           # Set sensor health for sensor type 2
hi = hi1 + hi2      # Calculate overall sensor health array
Rrs = Rrs1 + Rrs2   # Calculate overall sensor capacity array
wi = np.zeros(N)    # Weights initially set to zero and calculated later
ci = np.zeros(N)    # Cost Weights initially set to zero and calculated later
costs = np.zeros(N) # Costs initially set to zero and calculated based on values above
max_range = np.sqrt(((x_max-x_min) ** 2) + ((y_max-y_min) ** 2)) # Calculate the maximum range to cover entire simulation

# Calculate costs based on various parameters and use later on for the cost weights
for robot in range(1,N+1):
    costs[robot-1] = v_r[robot-1]
    if robot in N1:
        costs[robot-1] = (1/len(S)) * (hi1[list(N1).index(robot)] + Rrs1[list(N1).index(robot)])
    if robot in N2:
        costs[robot-1] = (1/len(S)) * (hi2[list(N2).index(robot)] + Rrs2[list(N2).index(robot)])

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

# Initialize figure
fig, ax = plt.subplots()

# Iterate for the amount defined
for k in range(iterations):

    # Get the poses of the robots and convert to single-integrator poses
    x = r.get_poses()
    x_si = uni_to_si_states(x)
    poses.append(x_si.tolist())
    current_x = x_si[0,:,None]
    current_y = x_si[1,:,None]

    # Instantiate calculation variables to zero
    Hg = 0
    Hp = 0
    Hv = 0
    Hr = 0
    Hc = 0
    c_v = np.zeros((N,2))
    w_v = np.zeros(N)
    w_vw = np.zeros(N)
    w_vc = np.zeros(N)
    cwi = np.zeros((2,N))
    distance_traveled = np.zeros(N)

    # Calculate each robot's sensor range
    ranges = np.array(Rrs) * max_range        

    # Nested loop that occurs for each coordinate, this is the calculation portion of the code for the Voronoi cells
    for ix in np.arange(x_min,x_max,res):
        for iy in np.arange(y_min,y_max,res):

            # Set the importance value to 1, this represents the distribution function
            importance_value = 1

            # Instantiate the distance array to zero for all values
            distances = np.zeros(N)

            # Calculate the distance of each robot to the current point
            for robots in range(N):
                distances[robots] = np.sqrt(np.square(ix - current_x[robots]) + np.square(iy - current_y[robots]))

            # Subtract the weights from distances for each robot
            weighted_distances = distances.copy()
            weighted_distances -= wi

            # Divide the velocities from distances for each robot
            velocity_distances = distances.copy()
            velocity_distances = velocity_distances/v_r

            # Subtract the costs from distances for each robot
            cost_distances = distances.copy()
            cost_distances -= ci

            # Find the robot with the smallest normal and weighted distances
            min_index = np.argmin(distances)
            weighted_min_index = np.argmin(weighted_distances)
            velocity_min_index = np.argmin(velocity_distances)
            cost_min_index = np.argmin(cost_distances)
            sensor_capacity = ranges[min_index]/2
            w_vw[weighted_min_index] += importance_value
            w_vc[cost_min_index] += importance_value

            # Intermediate calculations for the Standard Voronoi Partitioning
            c_v[min_index][0] += ix * importance_value
            c_v[min_index][1] += iy * importance_value
            w_v[min_index] += importance_value

            # Calculate costs
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
            Hc += 0.5 * ((distances[cost_min_index] ** 2) - ci[cost_min_index]) * importance_value  

    # Create a Voronoi diagram based on the robot positions
    points = np.array([current_x.flatten(), current_y.flatten()]).T
    vor = Voronoi(points)

    # Initialize the single-integrator control inputs
    si_velocities = np.zeros((2, N))

    # Make a copy of the current weights and costs, use this for calculations below
    weights = wi.copy()
    cost_weights = ci.copy()

    # Iterate for the number of robots, this is the velocity and weight controller portion of the code
    for robots in range(N):

        # Instantiate the summation variables to zero
        summation = 0
        cost_summation = 0

        # Calculate the summation for the weights controller
        for neighbor in range(N):
            summation += (hi[robots] - weights[robots]) - (hi[neighbor] - weights[neighbor])
            cost_summation += (costs[robots] - cost_weights[robots]) - (costs[neighbor] - cost_weights[neighbor])
        
        # Calculate the new weights
        wi[robots] += (kw/(2*w_vw[robots])) * summation
        ci[robots] += (kc/(2*w_vc[robots])) * cost_summation

        # Calculate the distance travled from the previous iteration
        if previous_x is not None and previous_y is not None:
            distance_traveled[robots] = float(np.sqrt((abs(current_x[robots][0] - previous_x[robots][0]) ** 2) + \
                                                      (abs(current_y[robots][0] - previous_y[robots][0]) ** 2)))

        # Calculate the x and y coordinates of the centroids of the Weighted Voronoi Partitions
        if not w_v[robots] == 0:
            c_x = c_v[robots][0] / w_v[robots]
            c_y = c_v[robots][1] / w_v[robots] 

            # Calcualte the velocity of each robot
            si_velocities[:, robots] = kv * np.array([(c_x - current_x[robots][0]), (c_y - current_y[robots][0])])

            # Append the current centroid and velocity to the lists
            cwi[:, robots] = np.array([c_x, c_y])

    # Make a copy of distance traveled array to calculate total distances
    total_distances = distance_traveled.copy()

    # If there is a value in the distance array, add the latest calculated distance to the previous iteration
    if dist_robot:
        total_distances += dist_robot[-1]

    # Update the variables for the previous pose with the current pose to be used in the next calculation
    previous_x = current_x
    previous_y = current_y

    # Add the current iteration values to the global lists 
    wij.append(wi.tolist())
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

    # Use the barrier certificate to avoid collisions
    si_velocities = si_barrier_cert(si_velocities, x_si)

    # Transform single integrator to unicycle
    dxu = si_to_uni_dyn(si_velocities, x)

    # Set the velocities of agents 1,...,N
    r.set_velocities(np.arange(N), dxu)
    # Iterate the simulation
    r.step()

    # Clear the previous plot
    ax.clear()

    # Plot the Voronoi diagram
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='r')

    # Plot robots' positions
    ax.scatter(current_x.flatten(), current_y.flatten(), c='b', marker='o')

    # Set plot limits and labels
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Iteration {k+1} - Voronoi Partitioning')

    # Draw the plot
    plt.draw()
    plt.pause(0.1)  # Pause to update the plot

# Outputs the data to a .csv file
with open("output.csv", mode="w", newline="") as file:
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
    writer.writerow(["Standard Voronoi Baseline"])

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()