import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

import rps.robotarium as robotarium
from rps.utilities.graph import *
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

# Ignore for now, trying to implement Weighted Voronoi Plotting
'''from geovoronoi import voronoi_regions_from_coords
from shapely.geometry import Polygon
from descartes import PolygonPatch'''

# Simulation Variables
N = 5                    # Number of robots
iterations = 500         # How many iterations do we want (about N*0.033 seconds)
L = lineGL(N)            # Generated a connected graph Laplacian (for a cylce graph).
x_min = -1.5             # Upper bound of x coordinate
x_max = 1.5              # Lower bound of x coordinate
y_min = -1               # Upper bound of y coordinate
y_max = 1                # Lower bound of x coordinate
res = 0.05               # Resolution of coordinates
convergence_threshold = 1e-3 
# Sensor Quality Variables
hi = np.ones(N)     # Sensor quality set to 1 initially for all sensors
hi[2] = 0.5         # Sensor quality of 3rd robot is 0.5
wi = np.zeros(N)    # Weights initially set to zero and calculated later
kw = 20            # Constant Gain for weights controller
k = 50               # Constant Gain for velocity controller

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


# We're working in single-integrator dynamics, and we don't want the robots
# to collide or drive off the testbed.  Thus, we're going to use barrier certificates
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

# Create SI to UNI dynamics tranformation
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# Initialize figure
fig, ax = plt.subplots()
previous_H_r = float('inf')
converged_iteration = -1
for k in range(iterations):

    # Get the poses of the robots and convert to single-integrator poses
    x = r.get_poses()
    x_si = uni_to_si_states(x)
    current_x = x_si[0,:,None]
    current_y = x_si[1,:,None]

    # Instantiate calculation variables to zero
    c_vw = np.zeros((N,2))
    w_vw = np.zeros(N)
    H = 0

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

            # Find the robot with the smallest weighted distance
            min_index = np.argmin(weighted_distances)

            # Intermeditae calculations for the Weighted Voronoi Partitioning
            c_vw[min_index][0] += ix * importance_value
            c_vw[min_index][1] += iy * importance_value
            w_vw[min_index] += importance_value

            # Add to the cost value
            H += 0.5 * ((distances[min_index] ** 2) - wi[min_index]) * importance_value
               
    # Print the cost and the Mass of the the Weighted Voronoi Regions
    print("Cost:", H)
    if abs(previous_H_r - H) < convergence_threshold:
        converged_iteration = k + 1
        print(f"Converged at iteration {converged_iteration} with H_p = {H}")
        break
    print("Mass of Voronoi:", w_vw)
    previous_H_r=H
    # Create a Voronoi diagram based on the robot positions
    points = np.array([current_x.flatten(), current_y.flatten()]).T
    vor = Voronoi(points)

    # Initialize the single-integrator control inputs
    si_velocities = np.zeros((2, N))
     
    neighbors = {i: set() for i in range(len(points))}

    # Extract neighbors from ridge points
    for p1, p2 in vor.ridge_points:
        neighbors[p1].add(p2)
        neighbors[p2].add(p1)

    # Convert sets to lists for easier use
    neighbors = {key: list(val) for key, val in neighbors.items()}

    # Make a copy of the current weights, use this for calculations below
    weights = wi.copy()

    # Iterate for the number of robots, this is the velocity and weight controller portion of the code
    for robots in range(N):

        # Instantiate the x and y coordinates of the centroids and the summation variable to zero
        c_x = 0
        c_y = 0
        summation = 0

        # Calculate weights based on sensor quality
        # for neighbor in neighbors[robots]:
        #     summation += hi[robots] - hi[int(neighbor)]

        # Calculate the summation for the weights controller
        for neighbor in range(N):
            summation += (hi[robots] - weights[robots]) - (hi[neighbor] - weights[neighbor])
        
        # Calculate the new weights
        wi[robots] += (kw/(2*w_vw[robots])) * summation

        # Calculate the x and y coordinates of the centroids of the Weighted Voronoi Partitions
        if not w_vw[robots] == 0:
            c_x = c_vw[robots][0] / w_vw[robots]
            c_y = c_vw[robots][1] / w_vw[robots] 

            # Calcualte the velcoity of each robot
            si_velocities[:, robots] = k * np.array([(c_x - current_x[robots][0]), (c_y - current_y[robots][0])])

    # Print the position of the 3rd robot and the weights, this robot has a value of 0.5 for Sensing Quality
    print("Current Pose of 3rd Robot:", current_x[2][0], current_y[2][0])
    print("Weights:", wi)

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

#Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()