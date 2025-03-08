import csv
import ast
import matplotlib.pyplot as plt

def plot_velocities(csv_file):

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

            # Extract the label
            elif counter == (10 * total_iterations + 21):
                label = row[0]

            counter += 1

    # Plot robot positions over iterations
    plt.figure(figsize=(8, 6))

    for robot_id, data in robot_velocities.items():
        plt.plot(data["x"], data["y"], linestyle="-", label=f"Robot {robot_id+1}")
        plt.scatter(data["x"][-1], data["y"][-1], marker="o", s=70)

    plt.xlabel("X Velocity (m/s)")
    plt.ylabel("Y Velocity (m/s)")
    plt.title("Robot Velocities Over Time")
    plt.suptitle(f"Scenario 1 - {label}")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    
    csv_files = ["output.csv", "output1.csv", "output2.csv", "output3.csv", "output4.csv", "output5.csv"]
    for csv_file in csv_files:
        plot_velocities(csv_file)

if __name__ == "__main__":
    main()