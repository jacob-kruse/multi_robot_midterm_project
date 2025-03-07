import csv
import ast
import matplotlib.pyplot as plt

def plot_poses(csv_file):

    iterations = []
    robot_centroids = {}
    label = None

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
            elif counter == (9 * total_iterations + 19):
                label = row[0]

            counter += 1

    # Plot robot positions over iterations
    plt.figure(figsize=(8, 6))

    for robot_id, data in robot_centroids.items():
        plt.plot(data["x"], data["y"], linestyle="-", label=f"Robot {robot_id+1}")
        plt.scatter(data["x"][-1], data["y"][-1], marker="o", s=10)

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.0, 1.0)
    plt.title("Robot Poses Over Time")
    plt.suptitle(f"Scenario 1 - {label}")
    plt.legend()
    plt.grid()
    plt.show()

def main():

    csv_files = ["output1.csv", "output2.csv", "output3.csv", "output4.csv"]
    for csv_file in csv_files:
        plot_poses(csv_file)

if __name__ == "__main__":
    main()