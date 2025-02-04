import numpy as np
import matplotlib.pyplot as plt


def y_function (x):
    return x**2

def y_derivitive (x):
    return 2*x

x = np.arange(-100,100,0.1)
y = y_function(x)
# print(x)
# print('-----------------')
# print(y)


current_posiotion1 = (80,y_function(80))
current_posiotion2 = (-80,y_function(-80))

learning_rate = 0.01

for _ in range(1000):
    new_x1 = current_posiotion1[0] - learning_rate * y_derivitive(current_posiotion1[0])
    new_y1 = y_function(new_x1)
    current_posiotion1=(new_x1,new_y1)

    new_x2 = current_posiotion2[0] - learning_rate * y_derivitive(current_posiotion2[0])
    new_y2 = y_function(new_x2)
    current_posiotion2 = (new_x2,new_y2)

    plt.plot(x,y)
    plt.scatter(current_posiotion1[0],current_posiotion1[1],color='red')
    plt.scatter(current_posiotion2[0],current_posiotion2[1],color='black')
    plt.pause(0.001)
    plt.clf()

