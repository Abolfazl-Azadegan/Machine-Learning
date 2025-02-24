import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(x)

# Here, x is a TensorFlow tensor with shape (2, 2) and data type float32.




# Load dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Check the shapes of the datasets
print("Training images shape:", train_images.shape)  # (50000, 32, 32, 3)
print("Training labels shape:", train_labels.shape)  # (50000, 1)
print("Test images shape:", test_images.shape)      # (10000, 32, 32, 3)
print("Test labels shape:", test_labels.shape)      # (10000, 1)

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images/ 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']


plt.figure(figsize=(5,5))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()