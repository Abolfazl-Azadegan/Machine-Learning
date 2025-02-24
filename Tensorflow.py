import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_PATH = "cifar10_data.npz"

# Check if dataset is already saved
if os.path.exists(DATA_PATH):
    # Load from saved file
    with np.load(DATA_PATH) as data:
        train_images, train_labels = data['train_images'], data['train_labels']
        test_images, test_labels = data['test_images'], data['test_labels']
    print("Dataset loaded from disk.")
else:
    # Load from TensorFlow dataset
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    print("Dataset downloaded and loaded into memory.")


    # Yes, every time you run the script, the dataset is reloaded, which means TensorFlow fetches it again from the local storage or 
    # downloads it if it's not already present.

    # However, TensorFlow automatically caches downloaded datasets in the ~/.keras/datasets/ directory (on Linux/Mac) or 
    # C:\Users\YourUsername\.keras\datasets\ (on Windows). So, even though the dataset is reloaded into memory on each run, 
    # it’s not re-downloaded from the internet every time.

    # Save to disk for future use
    np.savez(DATA_PATH, train_images=train_images, train_labels=train_labels, 
             test_images=test_images, test_labels=test_labels)
    print("Dataset saved to disk.")

# Normalize the images
train_images, test_images = train_images / 255.0, test_images / 255.0

# Display some images
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(5, 5))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()




# Get the default TensorFlow datasets directory
cache_dir = os.path.expanduser('~/.keras/datasets')  # For Linux/macOS
if os.name == 'nt':  # If running on Windows
    cache_dir = os.path.join(os.environ['USERPROFILE'], '.keras', 'datasets')

print("TensorFlow dataset cache directory:", cache_dir)

# List the files in the cache directory
if os.path.exists(cache_dir):
    print("Cached files:", os.listdir(cache_dir))
else:
    print("No cached datasets found.")



# How to Load the Saved Dataset?
# DATA_PATH = "cifar10_data.npz"

# # Load the dataset
# with np.load(DATA_PATH) as data:
#     train_images = data['train_images']
#     train_labels = data['train_labels']
#     test_images = data['test_images']
#     test_labels = data['test_labels']

# print("Dataset loaded from disk.")