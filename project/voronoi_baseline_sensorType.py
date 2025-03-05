import rps.robotarium as robotarium
from rps.utilities.graph import *
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import numpy as np

def get_sensor():
    """
    Function to return the sensor density measurement at a given location q.
    Currently returns a constant value of 1 (uniform distribution).
    """
    return 1

# Instantiate Robotarium object
N = 5  # Number of robots
ns = 2  # Number of sensor types
S = {1}  # Set of all sensor types
N1 = {1, 2, 3, 4, 5}  # Set of robots with sensor type 1
N2 = {}  # Set of robots with sensor type 2 (empty for now)
D_x, D_y = 100, 100  # Domain size in x and y directions

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

iterations = 500
convergence_threshold = 1e-3

si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()
L = lineGL(N)

x_min, x_max = -1.5, 1.5
y_min, y_max = -1, 1
res = 0.05
robot_sensors = {}

for i in range(N):
    robot_index = i + 1  
    assigned_sensors = set()
    if robot_index in N1:
        assigned_sensors.add(1)
    if robot_index in N2:
        assigned_sensors.add(2)
    robot_sensors[i] = assigned_sensors

fig, ax = plt.subplots()
previous_centroids = np.zeros((N, 2))
previous_H_g = float('inf')
converged_iteration = -1
for k in range(iterations):
    x = r.get_poses()
    x_si = uni_to_si_states(x)
    current_x = x_si[0,:,None]
    current_y = x_si[1,:,None]
    c_v = np.zeros((N, 2))
    w_v = np.zeros(N)

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

    si_velocities = np.zeros((2, N))

    for robot in range(N):
        sum_x, sum_y = 0, 0
        for j in robot_sensors[robot]:
            sensor_index = list(S).index(j)
            m_j_i = 0
            c_x_j, c_y_j = 0, 0
            
            # Calculate centroid and mass for each sensor type's Voronoi cell
            for ix in np.arange(x_min, x_max, res):
                for iy in np.arange(y_min, y_max, res):
                    q = np.array([ix, iy])
                    distances = np.full(N, np.inf)
                    for other_robot in range(N):
                        if j in robot_sensors[other_robot]:
                            dist = np.sqrt(np.square(ix - current_x[other_robot]) + np.square(iy - current_y[other_robot]))
                            distances[other_robot] = min(distances[other_robot], dist)
                    
                    min_index = np.argmin(distances)
                    if min_index == robot:
                        m_j_i += get_sensor()
                        c_x_j += ix * get_sensor()
                        c_y_j += iy * get_sensor()
            
            if m_j_i > 0:
                c_x_j /= m_j_i
                c_y_j /= m_j_i
                sum_x += m_j_i * (c_x_j - current_x[robot])
                sum_y += m_j_i * (c_y_j - current_y[robot])

        si_velocities[0, robot] = 2 * float(sum_x)
        si_velocities[1, robot] = 2 * float(sum_y)

    new_centroids = np.copy(previous_centroids) if previous_centroids is not None else np.zeros((N, 2))
    for robot in range(N):
        if w_v[robot] != 0:
            new_centroids[robot][0] = c_v[robot][0] / w_v[robot]
            new_centroids[robot][1] = c_v[robot][1] / w_v[robot]

    H_g = 0
    for j in S:
        for i in range(N):
            if j in robot_sensors[i]:
                integral_sum = 0
                for ix in np.arange(x_min, x_max, res):
                    for iy in np.arange(y_min, y_max, res):
                        q = np.array([ix, iy])
                        p_i = x_si[:, i]
                        d_ij = np.linalg.norm(p_i - q)**2
                        phi_j_q = get_sensor()  # Using get_sensor function
                        integral_sum += d_ij * phi_j_q * res * res
                H_g += integral_sum
    print(f"Iteration {k + 1}: H_g = {H_g}")

    if abs(previous_H_g - H_g) < convergence_threshold:
        converged_iteration = k + 1
        print(f"Converged at iteration {converged_iteration} with H_g = {H_g}")
        break
    previous_H_g = H_g

    previous_centroids = np.copy(new_centroids)
    #si_velocities = si_barrier_cert(si_velocities, x_si)
    dxu = si_to_uni_dyn(si_velocities, x)
    r.set_velocities(np.arange(N), dxu)
    r.step()

    points = np.array([current_x.flatten(), current_y.flatten()]).T
    vor = Voronoi(points)
    ax.clear()
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='r')
    ax.scatter(current_x.flatten(), current_y.flatten(), c='b', marker='o')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Iteration {k+1} - Voronoi Partitioning')
    plt.draw()
    plt.pause(0.1)

r.call_at_scripts_end()