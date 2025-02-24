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
    
    # The datasets.cifar10.load_data() function from TensorFlow's Keras module is used to load the CIFAR-10 dataset, which is a 
    # collection of 60,000 32x32 color images in 10 different classes. This function returns two tuples containing the training and 
    # test data, respectively.
    # What load_data() returns:
    # The function datasets.cifar10.load_data() returns two tuples:

    # Training data tuple: (train_images, train_labels)
    # Test data tuple: (test_images, test_labels)
    
    # What load_data() returns:
    # The function datasets.cifar10.load_data() returns two tuples:

    # Training data tuple: (train_images, train_labels)
    # Test data tuple: (test_images, test_labels)
    # Let’s explain the elements inside each of these tuples:

    # 1. train_images and test_images
    # These contain the image data for training and testing.
    # The images are 32x32 pixel images, and they are represented as NumPy arrays with the shape (num_images, 32, 32, 3) where:
    # num_images is the number of images in the dataset (50,000 for training and 10,000 for testing).
    # 32, 32 are the dimensions of the image (height and width).
    # 3 represents the three color channels (RGB), meaning each pixel has three values corresponding to Red, Green, and Blue.
    # Shape of train_images: (50000, 32, 32, 3)
    # Shape of test_images: (10000, 32, 32, 3)
    # 2. train_labels and test_labels
    # These contain the labels for the images.
    # Each label corresponds to a class (out of the 10 CIFAR-10 classes), and it’s stored as an integer. For example, the label 0 corresponds to "airplane", 1 to "automobile", and so on.
    # The labels are arrays with the shape (num_images, 1):
    # Shape of train_labels: (50000, 1) (one label for each training image)
    # Shape of test_labels: (10000, 1) (one label for each test image)
    
    # So, after running this line you will have:
    # train_images: A NumPy array with shape (50000, 32, 32, 3) containing the 50,000 training images.
    # train_labels: A NumPy array with shape (50000, 1) containing the labels for those 50,000 training images.
    # test_images: A NumPy array with shape (10000, 32, 32, 3) containing the 10,000 test images.
    # test_labels: A NumPy array with shape (10000, 1) containing the labels for those 10,000 test images.

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

plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i+1)
    # The function plt.subplot(3, 3, i+1) in Matplotlib is used to create a grid of subplots within a figure. Let's break down what 
    # it does:
    # plt.subplot(3, 3, i+1) explained:
    # 3: The first number represents the number of rows in the grid (3 rows in this case).
    # 3: The second number represents the number of columns in the grid (3 columns in this case).
    # i+1: The third number represents the index of the current subplot you're creating (starting from 1). i+1 is used to iterate
    # through each subplot in the grid. For example, if i is 0, it will create the first subplot, if i is 1, it will create the second 
    # subplot, and so on.
    # So, this line is dividing the figure into a 3x3 grid (9 subplots in total) and assigning each image to one of the grid positions.
    # The grid layout:
    # The grid will look something like this:
    # +-----+-----+-----+
    # |  1  |  2  |  3  |
    # +-----+-----+-----+
    # |  4  |  5  |  6  |
    # +-----+-----+-----+
    # |  7  |  8  |  9  |
    # +-----+-----+-----+
    # Where each number represents a subplot.
    # The i+1 ensures that the images are placed one by one in each of these positions. For example:
    # When i = 0, plt.subplot(3, 3, 1) will create the first subplot in the first position.
    # When i = 1, plt.subplot(3, 3, 2) will create the second subplot in the second position.
    # And so on until the 9th subplot.
    # Purpose of the subplot function:
    # The subplot function is used to create multiple plots in the same figure, helping you organize and compare different visualizations. You can control how the grid is arranged (in this case, a 3x3 grid) and place each individual plot in its specific location.
        
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
    
#     1. Understanding the Structure of train_labels
    # train_labels is a NumPy array, but each element in train_labels is a list or array containing a single number. The structure 
    # looks something like this:
    # train_labels = np.array([[3], [5], [7], ...])
    # In this case:
    # train_labels[0] = [3] (the first element is a list containing the number 3).
    # train_labels[1] = [5] (the second element is a list containing the number 5).
    # And so on.
    # The reason train_labels[i] is a list or array is because, in many cases, label arrays are structured this way 
    # (often for consistency or for cases where labels could potentially have multiple values or dimensions, though in this case, 
    # each list has only one element).
    # 2. Why Use [i][0] Instead of Just [i]
    # train_labels[i]
    # train_labels[i] gives us a list or array that contains the label for the i-th image. In the example:
    # train_labels[0] = [3]
    # train_labels[1] = [5]
    # So, if you access train_labels[i], you will get a list with a single value. For example:
    # train_labels[0] = [3]
    # train_labels[1] = [5]
    # The result of indexing train_labels[i] is not the number 3 or 5 directly, but a list containing that number.
    # train_labels[i][0]
    # When you use [i][0], you're first accessing the i-th element, which is a list (like [3] or [5]), and then using [0] to 
    # access the first element of that list (which is the actual label).
    # For example:
    # train_labels[0] = [3], and train_labels[0][0] = 3
    # train_labels[1] = [5], and train_labels[1][0] = 5
    # Why Not Just Use [i]?
    # If you use just train_labels[i], you'll get a list or array (e.g., [3] or [5]), not the numeric label.
    # By using train_labels[i][0], you are accessing the numeric value inside that list (e.g., 3 or 5).

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