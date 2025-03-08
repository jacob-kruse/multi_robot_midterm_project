import csv
import ast
import matplotlib.pyplot as plt

def plot_distances(csv_file):

    iterations = []
    robot_distances = {i: [] for i in range(5)}

    with open(csv_file, "r") as file:
        reader = csv.reader(file)

        counter = 0

        for row in reader:

            # Extract the number of iterations from the first row to use for next condition
            if counter == 0:
                total_iterations = int(row[0].split(':')[1].strip())

            # Extract the weights data from the corresponding rows
            if (4 * total_iterations + 9) < counter < (5 * total_iterations + 10):
                iteration = int(row[0].split()[1])  # Extract iteration number
                distances = ast.literal_eval(row[1])  # Convert string to list

                for i in range(len(distances)):
                    robot_distances[i].append(distances[i])

                iterations.append(iteration)

            # Extract the label
            elif counter == (10 * total_iterations + 21):
                label = row[0]

            counter += 1

    # Plot robot positions over iterations
    plt.figure(figsize=(8, 6))

    for robot_id, distances in robot_distances.items():
        plt.plot(iterations, distances, linestyle="-", label=f"Robot {robot_id+1}")

    plt.xlabel("Iteration")
    plt.ylabel("Distance Traveled (m)")
    plt.title("Total Distance Traveled")
    plt.suptitle(f"Scenario 1 - {label}")
    plt.legend()
    plt.grid()
    plt.show()

def main():

    csv_files = ["output.csv", "output1.csv", "output2.csv", "output3.csv", "output4.csv", "output5.csv"]
    for csv_file in csv_files:
        plot_distances(csv_file)

if __name__ == "__main__":
    main()