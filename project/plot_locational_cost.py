import csv
import matplotlib.pyplot as plt

def plot_locational_cost():
    # Read data from CSV file
    csv_file = "output.csv"  # Update with your actual file name

    iterations = []
    locational_costs = []

    with open(csv_file, "r") as file:
        reader = csv.reader(file)

        counter = 0

        for row in reader:

            # Extract the number of iterations from the first row to use for next condition
            if counter == 0:
                total_iterations = int(row[0].split(':')[1].strip())

            # Extract the locational cost data from the corresponding rows
            if (5 * total_iterations + 11) < counter < (6 * total_iterations + 12):
                iteration = int(row[0].split()[1].replace(':', ''))
                locational_cost = float(row[0].split(':')[1].strip())
                
                iterations.append(iteration)
                locational_costs.append(locational_cost)

            counter += 1

    # Plot robot positions over iterations
    plt.figure(figsize=(8, 6))

    plt.plot(iterations, locational_costs, linestyle="-")

    plt.xlabel("Iteration")
    plt.ylabel("Locational Cost")
    plt.title("Locational Cost over all Sensor Types")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    plot_locational_cost()

if __name__ == "__main__":
    main()