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

# Instantiate Robotarium object
N = 5
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

si_barrier_cert = create_single_integrator_barrier_certificate_with_boundary()
si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()
L = lineGL(N)

x_min, x_max, y_min, y_max, res = -1.5, 1.5, -1, 1, 0.05

previous_centroids = np.zeros((N, 2))

# Visualization setup
plt.ion()
fig, ax = plt.subplots()

for k in range(iterations):
    x = r.get_poses()
    x_si = uni_to_si_states(x)
    current_x = x_si[0, :, None]
    current_y = x_si[1, :, None]

    c_v = np.zeros((N, 2))
    w_v = np.zeros(N)

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
            si_velocities[:, robot] = 1 * (new_centroids[robot] - [current_x[robot, 0], current_y[robot, 0]])

    movement = np.linalg.norm(new_centroids - previous_centroids, axis=1)
    if np.all(movement < convergence_threshold):
        print(f"Converged at iteration {k+1}")
        break
    
    previous_centroids = np.copy(new_centroids)
    #si_velocities = si_barrier_cert(si_velocities, x_si)
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
    ax.set_title(f'Voronoi Partitioning - Iteration {k+1}')
    ax.legend()
    plt.pause(0.1)

plt.ioff()
plt.show()

r.call_at_scripts_end()
