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
iterations = 1000         # How many iterations do we want (about N*0.033 seconds)
L = lineGL(N)            # Generated a connected graph Laplacian (for a cylce graph).
x_min = -1.5             # Upper bound of x coordinate
x_max = 1.5              # Lower bound of x coordinate
y_min = -1               # Upper bound of y coordinate
y_max = 1                # Lower bound of x coordinate
res = 0.05               # Resolution of coordinates
convergence_threshold = 1e-3
# Sensor Capacity Variables
k = 50                        # Set the gain for the velocity controller
max_range = 2.0                 # Set the maximum range of ideal sensor
# max_range = np.sqrt(((x_max-x_min) ** 2) + ((y_max-y_min) ** 2)) # Calculate the maximum range to cover entire simulation
Rrs1 = np.ones(N)               # Sensor Capacity, set to 1 initially for all sensors
Rrs1[2] = 0.5                   # Set the Sensor Capacity of Robot 3 to 0.5

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
# Iterate for the amount defined
for k in range(iterations):

    # Get the poses of the robots and convert to single-integrator poses
    x = r.get_poses()
    x_si = uni_to_si_states(x)
    current_x = x_si[0,:,None]
    current_y = x_si[1,:,None]
 
    # points = np.zeros(shape=(int((x_max-x_min)/res),int((y_max-y_min)/res)))

    # Instantiate calculation variables to zero
    H = 0
    c_v = np.zeros((N,2))
    w_v = np.zeros(N)
    c_vr = np.zeros((N,2))
    w_vr = np.zeros(N)

    # Calculate each robot's sensor range
    ranges = Rrs1 * max_range      

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

            # Get the minimum distance, the closest robot, and calcualte the sensor capacity of the closest robot
            min_distance = min(distances)                 
            min_index = np.argmin(distances)
            sensor_capacity = ranges[min_index]/2

            # Intermeditae calculations for the Standard Voronoi Partitioning
            c_v[min_index][0] += ix * importance_value
            c_v[min_index][1] += iy * importance_value
            w_v[min_index] += 1

            # This is second part of the cost equation
            if min_distance > sensor_capacity:
                H += (sensor_capacity ** 2) * importance_value

            # This is the union of the regular Voronoi partition and range limited spherical radius
            # This is also the first part of the cost equation
            if min_distance <= sensor_capacity:

                c_vr[min_index][0] += ix * importance_value
                c_vr[min_index][1] += iy * importance_value
                w_vr[min_index] += 1

                H += (min_distance ** 2) * importance_value
   
    # Print the cost and the calculated Mass of each Ranged Voronoi Partition
    print("Cost:", H)
    if abs(previous_H_r - H) < convergence_threshold:
        converged_iteration = k + 1
        print(f"Converged at iteration {converged_iteration} with H_p = {H}")
        break
    print("Mass of Ranged Voronoi:", w_vr)
    previous_H_r = H        
    # Initialize the single-integrator control inputs
    si_velocities = np.zeros((2, N))

    # Iterate for the number of robots, this is the controller portion of the code
    for robots in range(N):

        # Instantiate the x and y coordinates of the centroids of the Standard and Ranged Voronoi Partitions to zero
        c_x = 0
        c_y = 0
        c_xr = 0
        c_yr = 0

        # Calculate the x and y coordinates of the centroids of the Standard and Ranged Voronoi Partitions
        if not w_v[robots] == 0:
            c_x = c_v[robots][0] / w_v[robots]
            c_y = c_v[robots][1] / w_v[robots]
            c_xr = c_vr[robots][0] / w_vr[robots]
            c_yr = c_vr[robots][1] / w_vr[robots]  
                   
            # Calcualte the velcoity of each robot
            si_velocities[:, robots] = k * np.array([(c_xr - current_x[robots][0]), (c_yr - current_y[robots][0] )])

    # Print the position of the 3rd robot, this robot has a value of 0.5 for Sensing Capacity
    print("Current Pose of 3rd Robot:", current_x[2][0], current_y[2][0])

    # Use the barrier certificate to avoid collisions
    si_velocities = si_barrier_cert(si_velocities, x_si)

    # Transform single integrator to unicycle
    dxu = si_to_uni_dyn(si_velocities, x)

    # Set the velocities of agents 1,...,N
    r.set_velocities(np.arange(N), dxu)

    # Iterate the simulation
    r.step()

    # Create a Voronoi diagram based on the robot positions
    points = np.array([current_x.flatten(), current_y.flatten()]).T
    vor = Voronoi(points)

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