import csv
import ast
import matplotlib.pyplot as plt

def plot_weights():
    # Read data from CSV file
    csv_file = "output.csv"  # Update with your actual file name

    iterations = []
    robot_weights = {i: [] for i in range(5)}

    with open(csv_file, "r") as file:
        reader = csv.reader(file)

        counter = 0

        for row in reader:

            # Extract the number of iterations from the first row to use for next condition
            if counter == 0:
                total_iterations = int(row[0].split(':')[1].strip())

            # Extract the weights data from the corresponding rows
            if (total_iterations + 3) < counter < (2 * total_iterations + 4):
                iteration = int(row[0].split()[1])  # Extract iteration number
                weights = ast.literal_eval(row[1])  # Convert string to list

                for i in range(len(weights)):
                    robot_weights[i].append(weights[i])

                iterations.append(iteration)

            counter += 1

    # Plot robot positions over iterations
    plt.figure(figsize=(8, 6))

    for robot_id, weights in robot_weights.items():
        plt.plot(iterations, weights, linestyle="-", label=f"Robot {robot_id+1}")

    plt.xlabel("Iteration")
    plt.ylabel("Weight")
    plt.title("Robot Weights")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    plot_weights()

if __name__ == "__main__":
    main()