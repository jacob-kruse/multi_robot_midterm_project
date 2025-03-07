import csv
import matplotlib.pyplot as plt

def plot_range_limited_cost():

    csv_files = ["output.csv", "output1.csv", "output2.csv", "output3.csv", "output4.csv", "output5.csv"]

    total_locational_costs = {}

    for csv_file in csv_files:
        
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
                elif (8 * total_iterations + 17) < counter < (9 * total_iterations + 18):
                    iteration = int(row[0].split()[1].replace(':', ''))
                    locational_cost = float(row[0].split(':')[1].strip())
                    
                    iterations.append(iteration)
                    locational_costs.append(locational_cost)

                # Extract the label
                elif counter == (10 * total_iterations + 21):
                    label = row[0]
                    if label not in total_locational_costs:
                        total_locational_costs[label] =  {"Costs": [], "Iterations": []}
                    total_locational_costs[label]["Costs"] = locational_costs
                    total_locational_costs[label]["Iterations"] = iterations

                counter += 1

    # Plot robot positions over iterations
    plt.figure(figsize=(8, 6))

    for label, data in total_locational_costs.items():
        plt.plot(data["Iterations"], data["Costs"], linestyle="-", label=f"{label}")
        plt.scatter(data["Iterations"][-1], data["Costs"][-1], marker="o", s=10)

    plt.xlabel("Iteration")
    plt.ylabel("Range-Limited Cost")
    plt.title("Range-Limited Cost")
    plt.suptitle("Scenario 1")
    plt.legend()
    plt.grid()
    plt.show()

def main():
    plot_range_limited_cost()

if __name__ == "__main__":
    main()