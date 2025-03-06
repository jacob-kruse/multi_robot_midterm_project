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
curvature_factor = 0.01  # Factor to add curvature to the movement
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
# Initialize new parameters
# Energy Availability (EA_r) updated based on health and sensor type
EA_r = [
    0.9 * hi1[0],  # Robot 1 (Sensor type 1)
    0.8 * hi1[1],  # Robot 2 (Sensor type 1)
    0.7 * hi1[2],  # Robot 3 (Sensor type 1)
    0.6 * hi2[1],  # Robot 4 (Sensor type 1)
    0.95 * hi2[2],  # Robot 5 (Sensor type 2)
    # 0.9 * hi2[1],   # Robot 6 (Sensor type 2)
    # 0.8 * hi2[2],   # Robot 7 (Sensor type 2)
    # 0.7 * hi2[3],   # Robot 8 (Sensor type 2)
    # 0.6 * hi2[4],   # Robot 9 (Sensor type 2)
    # 0.95 * hi2[4],  # Robot 10 (Sensor type 2)
]

# Service Capacity (SC_si) updated based on health and sensor type
SC_si = [
    0.9 * hi1[0],  # Robot 1 (Sensor type 1)
    0.85 * hi1[1],  # Robot 2 (Sensor type 1)
    0.8 * hi1[2],  # Robot 3 (Sensor type 1)
    0.75 * hi2[1],  # Robot 4 (Sensor type 1)
    0.95 * hi2[2],  # Robot 5 (Sensor type 2)
    # 0.9 * hi2[1],   # Robot 6 (Sensor type 2)
    # 0.85 * hi2[2],   # Robot 7 (Sensor type 2)
    # 0.8 * hi2[3],   # Robot 8 (Sensor type 2)
    # 0.75 * hi2[4],   # Robot 9 (Sensor type 2)
    # 0.95 * hi2[4],  # Robot 10 (Sensor type 2)
]
# Sensing Frequency (f_sri) updated based on health and velocity
#f_sri = [2.0, 1.8, 2.5, 1.5, 1.9, 2.0, 1.8, 2.5, 1.5, 1.9]  # Example sensing frequencies in Hz
f_sri = [2.0, 1.8, 2.5, 1.5, 1.9] 
f_sri_max = 2.0  # Max allowed sensing frequency for penalty calculation

f_sri = [
    min(f_sri_max, f_sri[0] * hi1[0]),  # Robot 1 (Sensor type 1)
    min(f_sri_max, f_sri[1] * hi1[1]),  # Robot 2 (Sensor type 1)
    min(f_sri_max, f_sri[2] * hi1[2]),  # Robot 3 (Sensor type 1)
    min(f_sri_max, f_sri[3] * hi2[1]),  # Robot 4 (Sensor type 1)
    min(f_sri_max, f_sri[4] * hi2[2]),  # Robot 5 (Sensor type 2)
    # min(f_sri_max, f_sri[5] * hi2[1]),  # Robot 6 (Sensor type 2)
    # min(f_sri_max, f_sri[6] * hi2[2]),  # Robot 7 (Sensor type 2)
    # min(f_sri_max, f_sri[7] * hi2[3]),  # Robot 8 (Sensor type 2)
    # min(f_sri_max, f_sri[8] * hi2[4]),  # Robot 9 (Sensor type 2)
    # min(f_sri_max, f_sri[9] * hi2[4]),  # Robot 10 (Sensor type 2)
]

# Field of View (fov_i) updated based on health and required FoV
#fov_i = [0.9, 0.8, 1.0, 0.85, 0.95, 0.9, 0.8, 1.0, 0.85, 0.95]  # Example field of view (normalized)
fov_i = [0.9, 0.8, 1.0, 0.85, 0.95]
#fov_required = [0.8, 0.85, 0.9, 0.75, 0.95, 0.8, 0.85, 0.9, 0.75, 0.95]  # Required FoV for each sensor type
fov_required = [0.8, 0.85, 0.9, 0.75, 0.95]
fov_i = [
    min(fov_required[0], fov_i[0] * hi1[0]),  # Robot 1 (Sensor type 1)
    min(fov_required[1], fov_i[1] * hi1[1]),  # Robot 2 (Sensor type 1)
    min(fov_required[2], fov_i[2] * hi1[2]),  # Robot 3 (Sensor type 1)
    min(fov_required[3], fov_i[3] * hi2[1]),  # Robot 4 (Sensor type 1)
    min(fov_required[4], fov_i[4] * hi2[2]),  # Robot 5 (Sensor type 2)
    # min(fov_required[5], fov_i[5] * hi2[1]),  # Robot 6 (Sensor type 2)
    # min(fov_required[6], fov_i[6] * hi2[2]),  # Robot 7 (Sensor type 2)
    # min(fov_required[7], fov_i[7] * hi2[3]),  # Robot 8 (Sensor type 2)
    # min(fov_required[8], fov_i[8] * hi2[4]),  # Robot 9 (Sensor type 2)
    # min(fov_required[9], fov_i[9] * hi2[4]),  # Robot 10 (Sensor type 2)
]
dynamic_weights = np.ones(7)  # Initialize dynamic weights
# Dynamically adjust weights for new parameters (low priority)
dynamic_weights[3] = 0.1  # Weight for energy
dynamic_weights[4] = 0.05  # Weight for service capacity
dynamic_weights[5] = 0.05  # Weight for sensing frequency 
dynamic_weights[6] = 0.05 # Weight for  FoV
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
def compute_cost(i, w_v, w_h, w_r, w_e, w_s, w_f):
    # Velocity cost (same as before)
    velocity_cost = Vr[i - 1] * w_v

    # Quality and Capacity cost calculation based on the robot type
    if i in N1:  # If robot i belongs to sensor type 1
        quality_cost = hi1[N1.index(i)] * w_h
        capacity_cost = Rrs1[N1.index(i)] * w_r
    else:  # If robot i belongs to sensor type 2
        quality_cost = hi2[N2.index(i)] * w_h
        capacity_cost = Rrs2[N2.index(i)] * w_r

    # Sensor type cost
    sensor_type_cost = (quality_cost + capacity_cost) / len(S)

    # Energy cost (add to the cost function based on energy availability)
    energy_cost = (1 - EA_r[i]) * w_e  # The lower the energy, the higher the cost

    # Service Capacity cost (add to the cost function based on service capacity)
    service_capacity_cost = (1 - SC_si[i]) * w_s  # The lower the service capacity, the higher the cost

    # Sensing Frequency cost (add based on sensing frequency)
    sensing_frequency_cost = max(0, f_sri[i] - f_sri_max) * w_f  # If frequency is lower than max, it's penalized

    # Field of View cost (penalize if the sensor's FoV is inadequate)
    fov_cost = max(0, fov_required[i] - fov_i[i]) * w_f  # Penalize if the FoV is not sufficient

    # Total cost with all factors
    total_cost = velocity_cost + sensor_type_cost + energy_cost + service_capacity_cost + sensing_frequency_cost + fov_cost
    return total_cost
iteration = 0

initial_adaptation_rate = 0.1
min_weights = np.array([0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05])  # Set minimums

def update_dynamic_weights():
    global dynamic_weights, iteration, w_v, w_h, w_r, EA_r, SC_si, f_sri, fov_i, initial_adaptation_rate, min_weights

    # Adaptive adaptation rate with smooth decay
    adaptation_rate = max(0.05, initial_adaptation_rate / (1 + iteration * 0.5))

    # Store the previous weights
    previous_weights = np.copy(dynamic_weights)
    print(f"Iteration {iteration + 1}, Adaptation Rate: {adaptation_rate}")
    print(f"Previous Weights: {previous_weights}")

    # Iterate over each robot
    for i in range(len(dynamic_weights)):
        # Initialize adjustments to be scalar values
        velocity_weight_adjustment = 0.0
        health_weight_adjustment = 0.0
        capacity_weight_adjustment = 0.0
        energy_weight_adjustment = 0.0
        service_capacity_weight_adjustment = 0.0
        sensing_frequency_weight_adjustment = 0.0
        field_of_view_weight_adjustment = 0.0

        # Scenario 1: Velocity adjustment (based on w_v)
        velocity_weight_adjustment = w_v * (1 - EA_r[i])  # Higher weight when energy is low

        # Scenario 2: Health adjustment (based on w_h)
        health_weight_adjustment = w_h * (1 - SC_si[i])  # Higher weight when health is low

        # Scenario 3: Capacity adjustment (based on w_r)
        capacity_weight_adjustment = w_r * (1 - SC_si[i])  # If service capacity is low, increase weight

        # Scenario 4: Energy availability adjustment (EA_r) and other factors
        energy_weight_adjustment = (1 - EA_r[i]) * 0.3  # Higher weight when energy is low

        # Scenario 5: Service capacity adjustment (SC_si)
        service_capacity_weight_adjustment = (1 - SC_si[i]) * 0.25  # If service capacity is low

        # Scenario 6: Sensing frequency adjustment (f_sri)
        sensing_frequency_weight_adjustment = (f_sri[i] - 1.0) * 0.15  # Penalize if the sensing frequency is higher

        # Scenario 7: Field of view adjustment (fov_i)
        field_of_view_weight_adjustment = (1 - fov_i[i]) * 0.2  # Higher weight when field of view is low

        # Calculate the total adjustment as a scalar value
        total_adjustment = (velocity_weight_adjustment + health_weight_adjustment +
                            capacity_weight_adjustment + energy_weight_adjustment +
                            service_capacity_weight_adjustment + sensing_frequency_weight_adjustment +
                            field_of_view_weight_adjustment)

        # Ensure each adjustment is scalar
        if isinstance(total_adjustment, np.ndarray):
            print(f"Error: Adjustment for robot {i + 1} is an array, not a scalar.")
            print(f"Adjustment: {total_adjustment}")
            return

        # Print out the individual adjustments for debugging
        print(f"Robot {i + 1} Adjustments:")
        print(f"  Velocity Adjustment: {velocity_weight_adjustment}")
        print(f"  Health Adjustment: {health_weight_adjustment}")
        print(f"  Capacity Adjustment: {capacity_weight_adjustment}")
        print(f"  Energy Adjustment: {energy_weight_adjustment}")
        print(f"  Service Capacity Adjustment: {service_capacity_weight_adjustment}")
        print(f"  Sensing Frequency Adjustment: {sensing_frequency_weight_adjustment}")
        print(f"  Field of View Adjustment: {field_of_view_weight_adjustment}")
        print(f"  Total Adjustment: {total_adjustment}")

        # Ensure dynamic_weights is a 1D numpy array of scalars
        print(f"Current Dynamic Weights (before update): {dynamic_weights}")
        
        # Update the dynamic weight for this robot (this should be scalar)
        dynamic_weights[i] += total_adjustment

    # Apply minimum thresholds to prevent collapse
    dynamic_weights = np.maximum(dynamic_weights, min_weights)

    # Stabilized normalization
    total_weight = np.sum(dynamic_weights)
    dynamic_weights = dynamic_weights / (total_weight + 0.1)  # Prevent collapse

    iteration += 1
    print("Updated Dynamic Weights:", dynamic_weights)
    return dynamic_weights


# Task allocation based on health and robot task complexity
def task_allocation(robot_health, N):
    assigned_tasks = []
    for i in range(N):
        if robot_health[i] > 0.75:  # Healthy robots are assigned complex tasks
            assigned_tasks.append(f"Robot {i+1} assigned to critical task")
        else:  # Weaker robots are assigned simple tasks
            assigned_tasks.append(f"Robot {i+1} assigned to less complex task")
    return assigned_tasks


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
    # Compute costs for each robot (before task allocation)
    costs = np.zeros(N)
    for i in range(1, N):
        costs[i] = compute_cost(i, dynamic_weights[0], dynamic_weights[1], dynamic_weights[2], dynamic_weights[3], dynamic_weights[4], dynamic_weights[5])
    print(f"Iteration {k+1} - Computed Costs: {costs}")
   
    # Assign tasks based on robot health
    assigned_tasks = task_allocation(hi1 + hi2, N)
    print(f"Assigned tasks: {assigned_tasks}")

    # Dynamically adjust weights before computing the cost
    update_dynamic_weights()

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
            si_velocities[:, robot] = k_p * (direction_to_target + curvature_adjustment / norm_direction) * dynamic_weights[0]  # Apply dynamic weight to velocity


    

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
