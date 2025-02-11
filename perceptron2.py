import numpy as np
import matplotlib.pyplot as plt
import random


def generate_training_set(num_points):
	x_coordinates = [random.randint(0, 50) for i in range(num_points)]
	y_coordinates = [random.randint(0, 50) for i in range(num_points)]
	training_set = dict()
	for x, y in zip(x_coordinates, y_coordinates):
		if x <= 45-y:
			training_set[(x,y)] = 1
		elif x > 45-y:
			training_set[(x,y)] = -1
	return training_set


# In a dictionary, you store data as key-value pairs.
# The line training_set[(x, y)] = 1 is assigning a value (1) to the key (x, y) in the training_set dictionary.
# If the key (x, y) does not exist, it adds a new entry.
# If the key (x, y) already exists, it overwrites the existing value with 1.


# The zip() function is a built-in Python utility that combines multiple iterables (like lists, tuples, or strings) into a single 
# iterable of tuples, where each tuple contains elements from the input iterables at the same position (index).

# Combines Elements by Index: It takes the first element from each iterable and makes a tuple, then the second element from each 
# iterable, and so on.
# Stops at the Shortest Iterable: If the iterables are of unequal length, zip() stops creating tuples when the shortest iterable is 
# exhausted.





training_set = generate_training_set(30)

x = np.arange(-10,60,0.1)
y = 45-x

x_plus = []

y_plus = []
x_minus = []
y_minus = []

for data in training_set:
	if training_set[data] == 1:
		x_plus.append(data[0])
		y_plus.append(data[1])
	elif training_set[data] == -1:
		x_minus.append(data[0])
		y_minus.append(data[1])

# data represents each key in the dictionary. Since the keys are tuples (x, y), data is a tuple in each iteration.



fig = plt.figure()
ax = plt.axes(xlim=(-10, 60), ylim=(-10, 60))


plt.scatter(x_plus, y_plus, marker = '+', c = 'green', s = 128, linewidth = 2)
plt.scatter(x_minus, y_minus, marker = '_', c = 'red', s = 128, linewidth = 2)

plt.title("Training Set")
plt.plot(x,y)
plt.show()