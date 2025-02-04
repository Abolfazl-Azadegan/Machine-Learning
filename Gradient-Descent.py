import numpy as np
import matplotlib.pyplot as plt


#
x = np.random.randn(10,1)
# print(x)
# print('--------------------------------------------------')
y = 2*x + np.random.rand(10,1)
# print(y)

# Here’s what happens:

# 2*x → Each value in x is multiplied by 2.
# np.random.rand() → Generates a single random number between 0 and 1.
# This single random number is added to all elements of 2*x.


#  If you want each element of y to have a different noise value, use:
# y = 2*x + np.random.rand(10,1)  # Now each element gets a different random value

# If you remove randomness, you can test how y = 2*x behaves:
# y = 2*x + 0.5  # Adds a fixed bias instead of a random one


w = 0.0
b = 0.0
learning_rate = 0.01

def descent (x,y,w,b,learning_rate):
    dldw=0.0
    dldb=0.0
    N=x.shape[0]
    for xi,yi in zip(x,y):
        #yhat=wx+b
        #loss=(y-yhat)**2/N
        #loss=(y-(wx+b))**2/N
        #loss=(y-wx-b)**2/N
        #dldw = (1/N)(2*U*dUdw) -> dldw = (2/N)(y-w*x-b)(-x) -> dldw = (-2/N)(x)(y-(w*x)-b)
        #dldb = (1/N)(2*U*dUdb) -> dldb = (2/N)(y-w*x-b)(-1) -> dldb = (-2/N)(y-(w*x)-b)
        dldw += (-2*xi*(yi-(w*xi+b)))/N
        dldb += (-2*(yi-(w*xi+b)))/N
    
    w = w - learning_rate*dldw
    b = b - learning_rate*dldb
    return w,b


# Create a plot and show it before the loop
plt.ion()  # Turn on interactive mode

for epoch in range(400):
    w , b = descent(x,y,w,b,learning_rate)
    yhat= w*x+b
    loss = np.divide(np.sum((y-yhat)**2, axis=0), x.shape[0])
    print(f'{epoch}: loss:{loss}, w:{w}, b:{b}')
    
    # Clear the previous plot
    plt.clf()
    
    # Plot original points
    plt.scatter(x, y, color='red', marker='o', label="Data Points")
    
    # Plot regression line
    x_line = np.linspace(np.min(x), np.max(x), 100).reshape(-1, 1)
    y_line = w * x_line + b
    plt.plot(x_line, y_line, color='blue', label=f"Epoch {epoch}")

    # Labels and title
    plt.xlabel("x (X-coordinates)")
    plt.ylabel("y (Y-coordinates)")
    plt.title(f"Linear Regression Progress (Epoch {epoch})")
    plt.legend()

    # Pause for visualization
    plt.pause(0.0001)

    print(f'{epoch}: loss:{loss}, w:{w}, b:{b}')

plt.ioff()  # Turn off interactive mode
plt.show()  # Show the final plot
 