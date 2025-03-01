import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import os
from IPython.display import Image

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


print(f"Number of training samples: {train_images.shape[0]}")
print(f"Number of test samples: {test_images.shape[0]}")

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



model = models.Sequential()
model.add(layers.Flatten(input_shape=(32, 32, 3)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
EPOCHS = 10
history = model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(test_images, test_labels))



# 1️⃣ Model Definition (Sequential)

# model = models.Sequential()
# What is happening here?
# We are creating a Sequential model using tf.keras.models.Sequential().
# Sequential means the model consists of a linear stack of layers (one layer after another).
# It is the simplest way to define a feedforward neural network.
# 2️⃣ Adding the First Layer (Flatten)

# model.add(layers.Flatten(input_shape=(32, 32, 3)))
# What is happening here?
# The Flatten() layer takes a 3D input (32 × 32 × 3 image) and converts it into a 1D vector.
# Each CIFAR-10 image is a color image of size 32×32 pixels with 3 color channels (RGB).
# So, a single image has a shape of (32, 32, 3) → Flatten turns it into a 1D array of size 32×32×3 = 3072.
# This does not change the data, it just reshapes it.
# 💡 Example:

# Before Flattening (3D)	After Flattening (1D)
# Image: 32 × 32 × 3	3072 values in a 1D array
# 3️⃣ Adding the First Dense (Fully Connected) Layer

# model.add(layers.Dense(64, activation='relu'))
# What is happening here?
# This is a Dense (Fully Connected) layer with 64 neurons.
# It takes the 3072 input values from the Flatten layer and connects them to 64 neurons.
# Each neuron applies a weighted sum of inputs + bias and then applies the ReLU activation function.
# Understanding activation='relu'
# ReLU (Rectified Linear Unit) function:
# 𝑓(𝑥)=max(0,𝑥)
# If the input is positive, it stays the same.
# If the input is negative, it becomes zero.
# This helps in faster training and avoids the vanishing gradient problem.
# Why use 64 neurons?
# The number of neurons is a hyperparameter (tunable).
# More neurons → More complex patterns learned.
# Fewer neurons → Faster training, but might not learn complex features.
# 4️⃣ Adding the Output Layer

# model.add(layers.Dense(10, activation='softmax'))
# What is happening here?
# This is the final layer of the model.
# It has 10 neurons because CIFAR-10 has 10 classes (airplane, automobile, bird, etc.).
# Each neuron represents one class.
# The activation function is softmax.
# Understanding activation='softmax'
# Softmax function converts raw output values into probabilities.
# It ensures that the sum of all 10 outputs = 1 (100%).
# The neuron with the highest probability is the predicted class.
# 💡 Example Output:

# Class	Raw Output	Softmax Output
# Airplane	3.1	0.70
# Car	2.0	0.20
# Bird	1.5	0.10
# Prediction: Airplane (since it has the highest probability: 0.70).

# 5️⃣ Displaying Model Summary

# model.summary()
# What is happening here?
# This prints a detailed summary of the model architecture, including:
# Layer types
# Number of parameters in each layer
# Shape of data as it passes through each layer
# 💡 Example Output of model.summary()

# Layer (type)         Output Shape     Param #  
# ==========================================
# Flatten (Flatten)    (None, 3072)     0
# Dense (Dense)        (None, 64)       196672
# Dense (Dense)        (None, 10)       650
# ==========================================
# Total params: 197,322
# Trainable params: 197,322
# Non-trainable params: 0
# 6️⃣ Compiling the Model

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# What is happening here?
# Before training, we need to specify:
# Optimizer: How the model updates weights.
# Loss function: How the model measures errors.
# Metrics: What performance measure to track.
# Understanding Each Parameter:
# optimizer='adam' 🏎️

# Adam (Adaptive Moment Estimation) is an adaptive learning rate optimizer.
# It adjusts learning rates automatically during training.
# It combines the benefits of SGD + RMSprop.
# loss='sparse_categorical_crossentropy' 🎯

# Since we have categorical labels (0-9), we use categorical cross-entropy.
# Since labels are integers, we use sparse_categorical_crossentropy.
# If labels were one-hot encoded, we would use categorical_crossentropy.
# metrics=['accuracy'] 📈

# Tracks accuracy during training & validation.
# 7️⃣ Training the Model

# EPOCHS = 10
# history = model.fit(train_images, train_labels, epochs=EPOCHS, validation_data=(test_images, test_labels))
# What is happening here?
# model.fit() trains the model using the training dataset.
# The model will run through the entire dataset 10 times (epochs=10).
# Each epoch consists of 1563 batches (each with 32 images).
# Understanding Each Parameter:
# train_images, train_labels → Training data (50,000 images).
# epochs=10 → The model will go through the full dataset 10 times.
# validation_data=(test_images, test_labels) → After each epoch, the model evaluates on test data.
# What happens during training?
# The model makes predictions on train_images.
# Computes the loss (error) between predicted & actual labels.
# Uses backpropagation + Adam optimizer to update weights.
# Repeats for all 10 epochs.
# 💡 Example Training Output:


# Epoch 1/10
# 1563/1563 ━━━━━━━━━━━━━━━━ 4s 2ms/step - accuracy: 0.26 - loss: 2.04 - val_accuracy: 0.34 - val_loss: 1.84
# Epoch 2/10
# 1563/1563 ━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.35 - loss: 1.81 - val_accuracy: 0.36 - val_loss: 1.76
# ...
# Epoch 10/10
# 1563/1563 ━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.45 - loss: 1.50 - val_accuracy: 0.42 - val_loss: 1.60
# What does this output mean?
# accuracy → Training accuracy on train_images.
# loss → Training loss (error).
# val_accuracy → Accuracy on test_images (validation set).
# val_loss → Loss on test_images.
# Final Summary
# Step-by-Step Explanation
# Step	Code	Purpose
# 1️⃣	Sequential()	Creates a stack of layers
# 2️⃣	Flatten()	Converts 32×32×3 images into 3072×1 vectors
# 3️⃣	Dense(64, activation='relu')	Adds 64 neurons (hidden layer) with ReLU
# 4️⃣	Dense(10, activation='softmax')	Adds 10 neurons for classification
# 5️⃣	compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])	Defines optimizer, loss function, and evaluation metric
# 6️⃣	fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))	Trains the model for 10 epochs







# Here's a breakdown of your understanding:
# Total Training Images = 50,000
# Epochs = 10 → This means the model will iterate over all 50,000 images 10 times.
# Batch Size = 32 → Instead of processing all 50,000 images at once, the model splits them into mini-batches of 32 images.
# Number of Batches per Epoch =
# 50000
# 32
# =
# 1562.5
# →
# rounded up to 1563
# 32
# 50000
# ​
#  =1562.5→rounded up to 1563
# So, in each epoch, the model processes 1563 batches, where each batch contains 32 images.
# This means that in one complete epoch, the model sees all 50,000 images once, but in smaller chunks of 32 images at a time.

# Final Summary:
# ✅ In one epoch, the model processes 1563 batches × 32 images per batch = ~50,000 images.
# ✅ In 10 epochs, the model will have seen each image 10 times in total.


def eval_metric(model, history, metric_name, EPOCHS):

    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]
    e = range(1, EPOCHS + 1)
    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.xlabel('Epoch number')
    plt.ylabel(metric_name)
    plt.title('Comparing training and validation ' + metric_name + ' for ' + model.name)
    plt.legend()
    plt.show()
    
eval_metric(model,history, 'loss',EPOCHS)

predictions = model.predict(test_images)
print(predictions[0]) 
np.argmax(predictions[0])



def plot_image(i, predictions_array, true_label, img):
    predictions_array, img = predictions_array, img[i]
    true_label_local = true_label[i][0]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label_local:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}%({})".format(class_names[predicted_label],100*np.max(predictions_array),class_names[true_label_local]),color=color)
    
def plot_value_array(i, predictions_array, true_label):
    true_label2 = true_label[i][0]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array,
    color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label2].set_color('green')
    
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()
    
    
i = 5
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()


