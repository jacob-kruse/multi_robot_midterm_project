import rps.robotarium as robotarium
from rps.utilities.graph import *
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np

# Instantiate Robotarium object
N = 5  # Number of robots
ns = 2  # Number of sensor types
S = {1}  # Set of all sensor types
N1 = {1, 2, 3, 4, 5}  # Set of robots with sensor type 1
N2 = {}  # Set of robots with sensor type 2 (empty for now)
D_x, D_y = 100, 100  # Domain size in x and y directions
phi_j = np.random.rand(N, len(S), D_x, D_y) 
r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)

# How many iterations do we want (about N*0.033 seconds)
iterations = 500
convergence_threshold = 1e-3  # Small threshold for movement

# Barrier certificates to avoid collisions
si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()

# Create SI to UNI dynamics transformation
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

# Generated a connected graph Laplacian (for a cycle graph)
L = lineGL(N)

# Environment boundaries
x_min = -1.5
x_max = 1.5
y_min = -1
y_max = 1
res = 0.05

# Dictionary to store robots' sensors
robot_sensors = {}

for i in range(N):
    robot_index = i + 1  # Robot index (starting from 1)
    assigned_sensors = set()
    
    # Assign sensor type 1 to robots in N1
    if robot_index in N1:
        assigned_sensors.add(1)
    
    # Assign sensor type 2 to robots in N2
    if robot_index in N2:
        assigned_sensors.add(2)
    
    # If a robot is in both N1 and N2, it will get both sensor types
    robot_sensors[i] = assigned_sensors

# Initialize the figure for plotting
fig, ax = plt.subplots()

# Initialize previous centroids
previous_centroids = np.zeros((N, 2))  # Initialize previous centroids

for k in range(iterations):
    # Get the poses of the robots and convert to single-integrator poses
    x = r.get_poses()
    x_si = uni_to_si_states(x)
    current_x = x_si[0,:,None]
    current_y = x_si[1,:,None]

    c_v = np.zeros((N, 2))
    w_v = np.zeros(N)

    # Compute Voronoi partitions based on sensor type
    for ix in np.arange(x_min, x_max, res):
        for iy in np.arange(y_min, y_max, res):
            importance_value = 1
            distances = np.full(N, np.inf)
            for robot in range(N):
                for sensor in robot_sensors[robot]:
                    dist = np.sqrt(np.square(ix - current_x[robot]) + np.square(iy - current_y[robot]))
                    distances[robot] = min(distances[robot], dist)
            min_index = np.argmin(distances)
            c_v[min_index][0] += ix * importance_value
            c_v[min_index][1] += iy * importance_value
            w_v[min_index] += importance_value

    # Initialize the single-integrator control inputs
    si_velocities = np.zeros((2, N))

    # Controller equation (Equation 15 from the paper)
    for robot in range(N):
        sum_x, sum_y = 0, 0  # Initialize sums for summation term in the equation

        for j in robot_sensors[robot]:  # Iterate over all sensor types for robot i
            # Compute m_j^i (mass contribution of sensor j for robot i)
            sensor_index = list(S).index(j)
            m_j_i = np.sum(phi_j[robot, sensor_index])  # Summing over all sensed points

            if m_j_i > 0:  # Avoid division by zero
                # Compute c_j^i (center of mass contribution)
                q_x, q_y = np.meshgrid(np.arange(D_x), np.arange(D_y))
                c_x = np.sum(q_x * phi_j[robot, sensor_index]) / m_j_i  # Weighted average x
                c_y = np.sum(q_y * phi_j[robot, sensor_index]) / m_j_i  # Weighted average y

                # Compute the summation term in the controller equation
                sum_x += m_j_i * (c_x - current_x[robot])
                sum_y += m_j_i * (c_y - current_y[robot])

    # Apply the control law for each robot
    si_velocities[0, robot] = 2 * sum_x
    si_velocities[1, robot] = 2 * sum_y

    # Compute new centroids
    new_centroids = np.copy(previous_centroids) if previous_centroids is not None else np.zeros((N, 2))
    for robot in range(N):
        if w_v[robot] != 0:
            new_centroids[robot][0] = c_v[robot][0] / w_v[robot]
            new_centroids[robot][1] = c_v[robot][1] / w_v[robot]

    # Compute locational cost H(p)
    H_p = 0
    for j in S:
        for i in range(N):
            if j in robot_sensors[i]:
                integral_sum = 0
                for ix in np.arange(x_min, x_max, res):
                    for iy in np.arange(y_min, y_max, res):
                        q = np.array([ix, iy])
                        d_ij = np.linalg.norm(q - x_si[:, i])**2
                        phi_j_q = 1  # Assuming uniform density function
                        integral_sum += d_ij * phi_j_q * res * res
                H_p += integral_sum
    print(f"Iteration {k + 1}: H_p = {H_p}")

    # Check for convergence
    if previous_centroids is not None:
        movement = np.linalg.norm(new_centroids - previous_centroids, axis=1)
        if np.all(movement < convergence_threshold):  # Check if all robots' movements are below the threshold
            print(f"Converged at iteration {k+1}")
            break

    previous_centroids = np.copy(new_centroids)

    # Use the barrier certificate to avoid collisions
    si_velocities = si_barrier_cert(si_velocities, x_si)

    # Transform single integrator to unicycle
    dxu = si_to_uni_dyn(si_velocities, x)

    # Set the velocities of agents 1,...,N
    r.set_velocities(np.arange(N), dxu)

    # Iterate the simulation
    r.step()

    # Plot Voronoi diagram
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

# Call at end of script to print debug information and for your script to run on the Robotarium server properly
r.call_at_scripts_end()
