import numpy as np
import matplotlib.pyplot as plt


def z_function(x,y):
    return np.sin(5 * x) * np.cos(5 * y) / 5


def calculate_gradient(x,y):
    return np.cos(5 * x) * np.cos(5 * y) , -np.sin(5 * x) * np.sin(5 * y)


x= np.arange(-1,1,0.05)
y= np.arange(-1,1,0.05)

X, Y = np.meshgrid(x,y)

Z = z_function(X,Y)

fig, ax = plt.subplots(subplot_kw={"projection": "3d","computed_zorder":False})

current_position1= (0.7 , 0.4 , z_function(0.7 , 0.4))
current_position2= (-0.2 , -0.4 , z_function(-0.2 , -0.4))
current_position3= (0.1 , -0.2 , z_function(0.1 , -0.2))
learning_rate = 0.01

for _ in range(1000):
    X_derivative1 , Y_derivative1 = calculate_gradient(current_position1[0],current_position1[1])
    X_new1 , Y_new1 = current_position1[0] - learning_rate * X_derivative1 , current_position1[1] - learning_rate * Y_derivative1
    current_position1 = (X_new1 , Y_new1 , z_function(X_new1,Y_new1))
 
    X_derivative2 , Y_derivative2 = calculate_gradient(current_position2[0],current_position2[1])
    X_new2 , Y_new2 = current_position2[0] - learning_rate * X_derivative2 , current_position2[1] - learning_rate * Y_derivative2
    current_position2 = (X_new2 , Y_new2 , z_function(X_new2,Y_new2))
    
    X_derivative3 , Y_derivative3 = calculate_gradient(current_position3[0],current_position3[1])
    X_new3 , Y_new3 = current_position3[0] - learning_rate * X_derivative3 , current_position3[1] - learning_rate * Y_derivative3
    current_position3 = (X_new3 , Y_new3 , z_function(X_new3 , Y_new3))
    
       
    
    ax.plot_surface(X,Y,Z,cmap='viridis',zorder=0)
    ax.scatter(current_position1[0],current_position1[1],current_position1[2],color="magenta",zorder=1)
    ax.scatter(current_position2[0],current_position2[1],current_position2[2],color="cyan",zorder=1)
    ax.scatter(current_position3[0],current_position3[1],current_position3[2],color="purple",zorder=1)
    plt.pause(0.001)
    ax.clear()