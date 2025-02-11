import matplotlib.pyplot as plt


def compute_output(w, x):
    z = 0.0
    for i in range(len(w)):
        print(f'x[{i}]:{x[i]} * w[{i}]:{w[i]}={x[i] * w[i]}')
        z += x[i] * w[i] # Compute sum of weighted inputs
    if z < 0: # Apply sign function
        return -1
    else:
        return 1
        

x1=[1,-1,-1]
x2=[1,1,-1]
x3=[1,-1,1]
x4=[1,1,1]

w=[0.9,-0.6,-0.5]

y1=compute_output(w,x1)
y2=compute_output(w,x2)
y3=compute_output(w,x3)
y4=compute_output(w,x4)


print(f'x1:{x1}, W:{w}, y1:{y1}')
print(f'x2:{x2}, W:{w}, y2:{y2}')
print(f'x3:{x3}, W:{w}, y3:{y3}')
print(f'x4:{x4}, W:{w}, y4:{y4}')



plt.scatter(x1, w, marker = '+', c = 'green', s = 128, linewidth = 2)
# plt.scatter(x2, w, marker = '_', c = 'red', s = 128, linewidth = 2)
# plt.scatter(x3, w, marker = '*', c = 'yellow', s = 128, linewidth = 2)
# plt.scatter(x4, w, marker = 'o', c = 'black', s = 128, linewidth = 2)
plt.show()


