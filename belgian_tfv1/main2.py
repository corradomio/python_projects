import os
import warnings
import silence_tensorflow as stf
import numpy as np
import random
import skimage
import skimage.io
import matplotlib.pyplot as plt

from skimage import transform
from skimage.color import rgb2gray
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import SGD

warnings.simplefilter(action='ignore', category=FutureWarning)
stf.silence_tensorflow()

ROOT_PATH = "D:/Projects/python/belgian_tfv1/BelgiumTSC"
train_data_directory = os.path.join(ROOT_PATH, "Training")
test_data_directory = os.path.join(ROOT_PATH, "Testing")
N_TRAINING_IMAGES = 4575  # max value: 4575
N_TESTING_IMAGES  = 2520  # max value 2520
N_OF_EPOCHS = 4001


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
# end


def load_images(filenames, labels, n_images=0):
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

    #
    # select images (and labels) specified by 'select'
    #
    if n_images != 0:
        select = select[0:n_images]  # select the first N_IMAGES elements
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

        # if len(images) % 20 == 0:
        #     print(".", end="")
        #     if len(images) % (1000) == 0:
        #         print("")
    # print("Done")

    return images, np.array(labels)
# end


def show_color_images(images, labels):
    plt.hist(labels, 62)
    plt.xlabel('categories')
    plt.ylabel('n images')
    plt.show()

    # Initialize the figure
    plt.figure(figsize=(15, 15))

    # Set a counter
    i = 1

    unique_labels = set(labels)

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
# end


def convert_images(images):
    # Rescale the images in the `images` array
    images28 = [transform.resize(image, (28, 28)) for image in images]

    # Convert `images28` to an array
    images28 = np.array(images28)

    # Convert `images28` to grayscale
    images28 = rgb2gray(images28)

    return images28
# end


def show_bw_images(images28, labels):
    # Initialize the figure
    plt.figure(figsize=(15, 15))

    # Set a counter
    i = 1

    unique_labels = set(labels)

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
# end


def create_and_fit_keras_nn(train_x, train_y, epochs=N_OF_EPOCHS):

    model_name = "model1"
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        # Dense(units=128, activation="relu"),
        # Dense(units=128, activation="relu"),
        Dense(units=128, activation="relu"),
        Dense(units=62, activation="softmax")
    ])

    model.compile(optimizer=SGD(.001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    print("fit with", len(train_x), "samples ..")
    model.fit(train_x, train_y, batch_size=32, epochs=epochs, verbose=2)

    fname = "models/{}_{}.hdf5".format(model_name, epochs)
    model.save(fname)

    print("done")
    return model


def main():
    #
    # Train
    #
    fnames, labels = read_filenames(train_data_directory)
    # max: 4575
    train_images, train_labels = load_images(fnames, labels, n_images=0)
    train_images28 = convert_images(train_images)

    model = create_and_fit_keras_nn(train_images28, train_labels, epochs=N_OF_EPOCHS)

    #
    # Test
    #
    fnames, labels = read_filenames(test_data_directory)
    test_images, test_labels = load_images(fnames, labels, n_images=0)
    test_images28 = convert_images(test_images)

    prediction = model.predict(test_images28)
    predicted_labels = np.argmax(prediction, axis=1)

    print("accuracy:", accuracy_score(test_labels, predicted_labels))
    pass
# end


if __name__ == "__main__":
    main()
