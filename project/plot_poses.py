import csv
import ast
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

def plot_poses(csv_file):

    iterations = []
    robot_centroids = {}
    label = None
    number_of_sensors = None

    with open(csv_file, "r") as file:
        reader = csv.reader(file)

        counter = 0

        for row in reader:

            # Extract the number of iterations from the first row to use for next condition
            if counter == 0:
                total_iterations = int(row[0].split(':')[1].strip())

            # Extract the pose data from the corresponding rows
            elif 1 < counter < (total_iterations + 2):
                iteration = int(row[0].split()[1])  # Extract iteration number
                positions = ast.literal_eval(row[1])  # Convert string to list

                x_positions, y_positions = positions  # Extract x and y lists

                # Store positions for each robot
                for i in range(len(x_positions)):
                    if i not in robot_centroids:
                        robot_centroids[i] = {"x": [], "y": []}
                    robot_centroids[i]["x"].append(x_positions[i])
                    robot_centroids[i]["y"].append(y_positions[i])

                iterations.append(iteration)

            # Extract the label
            elif counter == (10 * total_iterations + 21):
                label = row[0]

            # Extract the number of sensor types
            elif counter == (10 * total_iterations + 23):
                number_of_sensors = int(row[0])

            counter += 1

    # Plot robot positions over iterations
    fig, ax = plt.subplots(figsize=(8, 6))

    if len(list(robot_centroids)) == 5:
        if number_of_sensors == 1:
            final_x1 = np.zeros(5)
            final_y1 = np.zeros(5)
        elif number_of_sensors == 2:
            final_x1 = np.zeros(3)
            final_y1 = np.zeros(3)
            final_x2 = np.zeros(3)
            final_y2 = np.zeros(3)

    elif len(list(robot_centroids)) == 10:
        final_x1 = np.zeros(5)
        final_y1 = np.zeros(5)
        final_x2 = np.zeros(5)
        final_y2 = np.zeros(5)

    for robot_id, data in robot_centroids.items():
        plt.plot(data["x"], data["y"], linestyle="-", label=f"Robot {robot_id+1}")
        plt.scatter(data["x"][-1], data["y"][-1], marker="x", s=40)

        index = list(robot_centroids).index(robot_id)
        if len(list(robot_centroids)) == 5:
            if number_of_sensors == 1:
                final_x1[index] = data["x"][-1]
                final_y1[index] = data["y"][-1]
            elif number_of_sensors == 2:
                if index <= 2:
                    final_x1[index] = data["x"][-1]
                    final_y1[index] = data["y"][-1]
                if index >= 2:
                    final_x2[index-3] = data["x"][-1]
                    final_y2[index-3] = data["y"][-1]
        elif len(list(robot_centroids)) == 10:
            if index <= 4:
                final_x1[index] = data["x"][-1]
                final_y1[index] = data["y"][-1]
            if index >= 4:
                final_x2[index-5] = data["x"][-1]
                final_y2[index-5] = data["y"][-1]

    final_points1 = np.array([final_x1.flatten(), final_y1.flatten()]).T
    vor1 = Voronoi(final_points1)
    voronoi_plot_2d(vor1, ax=ax, show_points=False, show_vertices=False, line_colors='#000080', line_width=2)
    if (len(list(robot_centroids)) == 5 and number_of_sensors == 2) or len(list(robot_centroids)) == 10:
        final_points2 = np.array([final_x2.flatten(), final_y2.flatten()]).T
        vor2 = Voronoi(final_points2)
        voronoi_plot_2d(vor2, ax=ax, show_points=False, show_vertices=False, line_colors='#8B0055', line_width=2)

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.0, 1.0)
    ax.set_title("Robot Poses Over Time")
    fig.suptitle(f"Scenario 1 - {label}")
    # ax.legend()
    ax.grid()
    plt.show()

def main():

    csv_files = ["output.csv", "output1.csv", "output2.csv", "output3.csv", "output4.csv", "output5.csv"]
    for csv_file in csv_files:
        plot_poses(csv_file)

if __name__ == "__main__":
    main()