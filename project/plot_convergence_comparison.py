import csv
import matplotlib.pyplot as plt

def plot_convergence_comparison():

    csv_files = ["output1.csv", "output2.csv", "output3.csv", "output4.csv"]

    convergences = {}

    for csv_file in csv_files:

        with open(csv_file, "r") as file:
            reader = csv.reader(file)

            counter = 0

            for row in reader:

                # Extract the number of iterations from the first row to use for next condition
                if counter == 0:
                    total_iterations = int(row[0].split(':')[1].strip())

                # Extract the label
                elif counter == (9 * total_iterations + 19):
                    label = row[0]
                    if label not in convergences:
                        convergences[label] =  {"Iteration": int}
                    convergences[label]["Iteration"] = total_iterations

                counter += 1

    # Plot robot positions over iterations
    plt.figure(figsize=(8, 6))

    for label, data in convergences.items():
        bar = plt.bar(label, data["Iteration"])
        yval = int(bar[0].get_height())
        plt.text(bar[0].get_x() + bar[0].get_width() / 2, yval, f'{yval}', ha='center', va='bottom', fontsize=10)

    # plt.xlabel("Iteration")
    plt.ylabel("Iteration")
    plt.title("Iteration of Convergence")
    plt.suptitle("Scenario 1")
    plt.xticks(fontsize=8)
    plt.show()

def main():
    plot_convergence_comparison()

if __name__ == "__main__":
    main()