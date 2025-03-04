import csv
import ast
import matplotlib.pyplot as plt

def plot_velocities():
    # Read data from CSV file
    csv_file = "output.csv"  # Update with your actual file name

    iterations = []
    robot_velocities = {}

    with open(csv_file, "r") as file:
        reader = csv.reader(file)

        counter = 0

        for row in reader:

            # Extract the number of iterations from the first row to use for next condition
            if counter == 0:
                total_iterations = int(row[0].split(':')[1].strip())

            # Extract the weights data from the corresponding rows
            if (3 * total_iterations + 7) < counter < (4 * total_iterations + 8):
                iteration = int(row[0].split()[1])  # Extract iteration number
                positions = ast.literal_eval(row[1])  # Convert string to list

                x_positions, y_positions = positions  # Extract x and y lists

                # Store positions for each robot
                for i in range(len(x_positions)):
                    if i not in robot_velocities:
                        robot_velocities[i] = {"x": [], "y": []}
                    robot_velocities[i]["x"].append(x_positions[i])
                    robot_velocities[i]["y"].append(y_positions[i])

                iterations.append(iteration)

            counter += 1

    # Plot robot positions over iterations
    plt.figure(figsize=(8, 6))

    for robot_id, data in robot_velocities.items():
        plt.plot(data["x"], data["y"], linestyle="-", label=f"Robot {robot_id+1}")
        plt.scatter(data["x"][-1], data["y"][-1], marker="o", s=70)

    plt.xlabel("X Velocity (m/s)")
    plt.ylabel("Y Velocity (m/s)")
    plt.title("Robot Velocities Over Time")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    plot_velocities()

if __name__ == "__main__":
    main()