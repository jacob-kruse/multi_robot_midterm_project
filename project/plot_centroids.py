import csv
import ast
import matplotlib.pyplot as plt

def plot_centroids():
    # Read data from CSV file
    csv_file = "output.csv"  # Update with your actual file name

    iterations = []
    robot_positions = {}

    with open(csv_file, "r") as file:
        reader = csv.reader(file)

        counter = 0

        for row in reader:

            # Extract the number of iterations from the first row to use for next condition
            if counter == 0:
                total_iterations = int(row[0].split(':')[1].strip())

            # Extract the centroid data from the corresponding rows
            if (2 * total_iterations + 5) < counter < (3 * total_iterations + 6):
                iteration = int(row[0].split()[1])  # Extract iteration number
                positions = ast.literal_eval(row[1])  # Convert string to list

                x_positions, y_positions = positions  # Extract x and y lists

                # Store positions for each robot
                for i in range(len(x_positions)):
                    if i not in robot_positions:
                        robot_positions[i] = {"x": [], "y": []}
                    robot_positions[i]["x"].append(x_positions[i])
                    robot_positions[i]["y"].append(y_positions[i])

                iterations.append(iteration)

            counter += 1

    # Plot robot positions over iterations
    plt.figure(figsize=(8, 6))

    for robot_id, data in robot_positions.items():
        plt.plot(data["x"], data["y"], linestyle="-", label=f"Robot {robot_id+1}")
        plt.scatter(data["x"][-1], data["y"][-1], marker="o", s=10)

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.0, 1.0)
    plt.title("Robot Centroids Over Time")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    plot_centroids()

if __name__ == "__main__":
    main()