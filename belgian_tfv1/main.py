import os

import numpy as np
import random
import skimage
import skimage.io


ROOT_PATH = "D:/Projects/python/belgian_tfv1/BelgiumTSC"
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")

print(train_data_directory, test_data_directory)

print("Reading image files and labels ...")


def read_filenames(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    file_images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            file_images.append(f)
            labels.append(int(d))
    return file_images, labels


filenames, labels = read_filenames(train_data_directory)
print("Found {} images".format(len(filenames)))

import numpy as np
import random
import skimage
import skimage.io


def load_images(filenames, labels, n_images):
    """
    :param filenames: list of all image files
    :param labels:    list of all image labels
    :param n_images:  n of  images to load
    """
    #
    # generate a random permutation of numbers [0,1,...4575-1]
    # and select the first 'n_images' indices
    #
    n_files = len(filenames)  # number of images
    select = list(range(n_files))  # generate the list [0,1,2,..,4574]
    random.shuffle(select)  # shuffle the list
    select = select[0:n_images]  # select the first N_IMAGES elements

    #
    # select images (and labels) specified by 'select'
    #
    filenames = [filenames[i] for i in select]
    labels = [labels[i] for i in select]

    #
    # load the images in memory
    #
    print("Loading images ...")
    images = []
    for f in filenames:
        image = skimage.io.imread(f)
        images.append(image)

        print(".", end="")
        if len(images) % 100 == 0:
            print("")
    print("Done")
    return images, labels
# end

print(load_images)

N_TRAINING_IMAGES = 4575  # max value: 4575
# each row are 100 images

images, labels = load_images(filenames, labels, N_TRAINING_IMAGES)
print(len(images), len(labels))

import matplotlib.pyplot as plt

# Make a histogram with 62 bins of the `labels` data
plt.hist(labels, 62)
plt.xlabel('categories')
plt.ylabel('n images')
plt.show()

import matplotlib.pyplot as plt

# Get the unique labels
unique_labels = set(labels)

# Initialize the figure
plt.figure(figsize=(15, 15))

# Set a counter
i = 1

# For each unique label,
for label in unique_labels:
    # You pick the first image for each label
    image = images[labels.index(label)]
    # Define 64 subplots
    plt.subplot(8, 8, i)
    # Don't include axes
    plt.axis('off')
    # Add a title to each subplot
    # plt.title("Label {0} ({1})".format(label, labels.count(label)))
    # Add 1 to the counter
    i += 1
    # And you plot this first image
    plt.imshow(image, interpolation='nearest')

# Show the plot
plt.show()

# Import the `transform` module from `skimage`
from skimage import transform
# Import `rgb2gray` from `skimage.color`
from skimage.color import rgb2gray

print("Converting images ...")

# Rescale the images in the `images` array
images28 = [transform.resize(image, (28, 28)) for image in images]

# Convert `images28` to an array
images28 = np.array(images28)

# Convert `images28` to grayscale
images28 = rgb2gray(images28)

print("Done")

# Initialize the figure
plt.figure(figsize=(15, 15))

# Set a counter
i = 1

# For each unique label,
for label in unique_labels:
    # You pick the first image for each label
    image = images28[labels.index(label)]
    # Define 64 subplots
    plt.subplot(8, 8, i)
    # Don't include axes
    plt.axis('off')
    # Add a title to each subplot
    # plt.title("Label {0} ({1})".format(label, labels.count(label)))
    # Add 1 to the counter
    i += 1
    # And you plot this first image
    plt.imshow(image, interpolation='nearest', cmap='gray')

# Show the plot
plt.show()

import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("Structure of the NeuralNetwork ...")

# Initialize placeholders
x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28])
y = tf.placeholder(dtype=tf.int32, shape=[None])

# Flatten the input data
images_flat = tf.contrib.layers.flatten(x)

# Fully connected layer
logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))

# Define an optimizer
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("  images_flat: ", images_flat)
print("  logits: ", logits)
print("  loss: ", loss)
print("  predicted_labels: ", correct_pred)

print("Done")

import tensorflow as tf

tf.set_random_seed(1234)

N_OF_EPOCHS = 4001

listOfLosses = []
global_sess = tf.Session()
global_sess.run(tf.global_variables_initializer())

for i in range(N_OF_EPOCHS):
    _, loss_value = global_sess.run([train_op, loss], feed_dict={x: images28, y: labels})
    listOfLosses.append(loss_value)
    if i % 500 == 0:
        print("epoch: {:4}, loss: {}".format(i, loss_value))

# Pick 10 random images
sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

# Run the "correct_pred" operation
predicted = global_sess.run([correct_pred], feed_dict={x: sample_images})[0]
print("Done")

# Print the real and predicted labels
print('ground truth', list(sample_labels))
print('prediction  ', list(predicted))

import matplotlib.pyplot as plt

plt.plot(listOfLosses)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

import matplotlib.pyplot as plt

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2, 1 + i)
    plt.axis('off')
    color = 'green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i], cmap="gray")

plt.show()

N_TESTING_IMAGES = 200  # max value 2520

test_filenames, test_labels = read_filenames(test_data_directory)

test_images, test_labels = load_images(test_filenames, test_labels, N_TESTING_IMAGES)

print(len(test_images), len(test_labels))

from skimage import transform
from skimage.color import rgb2gray

print("Converting images ...")

# Transform the images to 28 by 28 pixels
test_images28 = [transform.resize(image, (28, 28)) for image in test_images]

# Convert to grayscale
test_images28 = rgb2gray(np.array(test_images28))

print("Done")

# Run predictions against the full test set.
predicted = global_sess.run([correct_pred], feed_dict={x: test_images28})[0]

# Calculate correct matches
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))
print('ground truth', list(test_labels[0:10]))
print('predicted   ', list(predicted)[0:10])

import matplotlib.pyplot as plt

# Pick 10 random images
sample_images = test_images28[0:10]
sample_labels = test_labels[0:10]

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2, 1 + i)
    plt.axis('off')
    color = 'green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i], cmap="gray")

plt.show()
