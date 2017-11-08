import csv
import matplotlib.pyplot as plt


# Load a CSV file
def load_csv(filename):
    #init the dataset as a list
	dataset = list()
    #open it as a readable file
	with open(filename, 'r') as file:
        #init the csv reader
		csv_reader = csv.reader(file)
        #for every row in the dataset
		for row in csv_reader:
            #add that row as an element in our dataset list (2D Matrix of values)
			dataset.append(row)
    #return in-memory data matrix
	return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    #iterate throw all the rows in our data matrix
	for row in dataset:
        #for the given column index, convert all values in that column to floats
		row[column] = float(row[column].strip())


def main():

    avg_reward = list()
    avg_reward_decay = list()

    avg_reward = load_csv('constant_avg')
    # convert string attributes to integers
    for i in range(0, len(avg_reward[0])):
        str_column_to_float(avg_reward, i)
    avg_reward = avg_reward[0]

    avg_reward_decay = load_csv('decay_avg')
    # convert string attributes to integers
    for i in range(0, len(avg_reward_decay[0])):
        str_column_to_float(avg_reward_decay, i)
    avg_reward_decay = avg_reward_decay[0]

    timesteps = range(1,len(avg_reward)+1)
    timesteps_decay = range(1,len(avg_reward_decay)+1)
    plt.plot(timesteps,avg_reward,'r',timesteps_decay,avg_reward_decay,'b')
    plt.ylabel("Average reward")
    plt.xlabel("Timesteps")
    plt.legend(["constant exploration rate","decaying exploration rate"])
    plt.show()


if __name__ == '__main__':
    main()