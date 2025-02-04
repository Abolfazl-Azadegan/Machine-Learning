import numpy as np
import matplotlib.pyplot as plt


num = np.random.randn()
num2 = np.random.randn(5)
num4 = np.random.randn(5,1)
num3 = np.random.rand(2,3)
print(num)
print(num2)
print(num4)
# print(num3)
print(np.ndim(num2))
print(np.ndim(num4))
print(np.shape(num2))
print(np.shape(num4))

data = np.random.randn(10000)  # Generate 10,000 numbers
plt.hist(data, bins=50, density=True, alpha=0.6, color='b')
plt.title("Histogram of np.random.randn()")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


# Convert (5,) to (5,1) using .reshape(-1, 1):
arr1_reshaped = arr1.reshape(-1, 1)  # Now shape (5,1)
# Convert (5,1) to (5,) using .flatten() or .ravel():  
arr2_flattened = arr2.flatten()  # Now shape (5,)
# (5,) is a 1D array (vector), which is not explicitly a row or column.
# (5,1) is a 2D column vector, explicitly structured for matrix operations.
# (5,1) → A 2D Column Vector
# It has 5 rows and 1 column.
# It is explicitly a column vector in linear algebra.
# It can be used in matrix multiplication with row vectors.


# Difference Between     (5,)      and      (1,5)
# Shape	                 (5,) (1D Vector)	(1,5) (Row Vector)
# Example	             [1, 2, 3, 4, 5]	[[1, 2, 3, 4, 5]]
# Dimensions	         1D array	        2D row vector
# Shape	                 (5,)	            (1,5)
# Matrix Operations	     Limited	        Works in matrix multiplication

# Convert a 1D array to a Row Vector (1,5)
arr = np.array([1, 2, 3, 4, 5])  # Shape (5,)
row_vec = arr.reshape(1, -1)  # Shape (1,5)

print(row_vec)
print("Shape:", row_vec.shape)  # Output: (1, 5)



# Convert a Row Vector (1,5) to a 1D array (5,)
flattened = row_vec.flatten()  # or row_vector.ravel()
print(flattened)
print("Shape:", flattened.shape)  # Output: (5,)


# Using a Row Vector in Matrix Multiplication
col_vector = np.array([[1], [2], [3], [4], [5]])  # Shape (5,1)

# Row vector (1,5) @ Column vector (5,1) → Result: (1,1) matrix
result = row_vec @ col_vector

print(result)
print("Shape:", result.shape)  # Output: (1,1)


