from plot_poses import plot_poses
from plot_weights import plot_weights
from plot_centroids import plot_centroids
from plot_velocities import plot_velocities
from plot_distances import plot_distances
from plot_locational_cost_comparison import plot_locational_cost
from plot_power_cost_comparison import plot_power_cost
from plot_temporal_cost_comparison import plot_temporal_cost
from plot_range_limited_cost_comparison import plot_range_limited_cost
from plot_convergence_comparison import plot_convergence_comparison
from plot_custom_heterogeneous_cost_comparison import plot_custom_heterogeneous_cost

csv_files = ["output.csv", "output1.csv", "output2.csv", "output3.csv", "output4.csv", "output5.csv"]
for csv_file in csv_files:
    plot_poses(csv_file)
plot_weights()
plot_centroids()
plot_velocities()
plot_distances()
plot_locational_cost()
plot_power_cost()
plot_temporal_cost()
plot_range_limited_cost()
plot_custom_heterogeneous_cost()
plot_convergence_comparison()