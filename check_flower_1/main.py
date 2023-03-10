import sys

import numpy as np
import random
import cv2
import os
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
from keras import backend as K

from fl_mnist_implementation_tutorial_utils import *


print(sys.version)


def load(paths, verbose=-1):
    '''expects images for each class in seperate dir,
    e.g all digits in 0 class in the directory named 0 '''
    data = list()
    labels = list()
    # loop over the input images
    for (i, imgpath) in enumerate(paths):
        # load the image and extract the class labels
        im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        image = np.array(im_gray).flatten()
        label = imgpath.split(os.path.sep)[-2]
        # scale the image to [0, 1] and add to list
        data.append(image/255)
        labels.append(label)
        # show an update every `verbose` images
        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))
    # return a tuple of the data and labels
    return data, labels

#declear path to your mnist data folder
img_path = 'mnist/trainingSet/trainingSet'

#get the path list using the path object
image_paths = list(paths.list_images(img_path))

#apply our function
image_list, label_list = load(image_paths, verbose=10000)

#binarize the labels
lb = LabelBinarizer()
label_list = lb.fit_transform(label_list)

#split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.1, random_state=42)


def create_clients(image_list, label_list, num_clients=10, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as
                data shards - tuple of images and label lists.
        args:
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1

    '''

    # create a list of client names
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]

    # randomize the data
    data = list(zip(image_list, label_list))
    random.shuffle(data)

    # shard data and place at each client
    size = len(data) // num_clients
    shards = [data[i:i + size] for i in range(0, size * num_clients, size)]

    # number of clients must equal number of shards
    assert (len(shards) == len(client_names))

    return {client_names[i]: shards[i] for i in range(len(client_names))}


# create clients
clients = create_clients(X_train, y_train, num_clients=10, initial='client')


def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    # seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)


# process and batch the training data for each client
clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)

# process and batch the test set
test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))


class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model


lr = 0.01
comms_round = 100
loss = 'categorical_crossentropy'
metrics = ['accuracy']
optimizer = SGD(learning_rate=lr, decay=lr / comms_round, momentum=0.9)


def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    # get the bs
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    # first calculate the total training data points across clinets
    global_count = sum(
        [tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names]) * bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() * bs
    return local_count / global_count


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    return avg_grad


def test_model(X_test, Y_test, model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss


# initialize global model
smlp_global = SimpleMLP()
global_model = smlp_global.build(784, 10)

# commence global training loop
for comm_round in range(comms_round):

    # get the global model's weights - will serve as the initial weights for all local models
    global_weights = global_model.get_weights()

    # initial list to collect local model weights after scalling
    scaled_local_weight_list = list()

    # randomize client data - using keys
    client_names = list(clients_batched.keys())
    random.shuffle(client_names)

    # loop through each client and create new local model
    for client in client_names:
        smlp_local = SimpleMLP()
        local_model = smlp_local.build(784, 10)
        local_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        # set local model weight to the weight of the global model
        local_model.set_weights(global_weights)

        # fit local model with client's data
        local_model.fit(clients_batched[client], epochs=1, verbose=0)

        # scale the model weights and add to list
        scaling_factor = weight_scalling_factor(clients_batched, client)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)

        # clear session to free memory after each communication round
        K.clear_session()

    # to get the average over all the local model, we simply take the sum of the scaled weights
    average_weights = sum_scaled_weights(scaled_local_weight_list)

    # update global model
    global_model.set_weights(average_weights)

    # test global model and print out metrics after each communications round
    for (X_test, Y_test) in test_batched:
        global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)


def test_model(X_test, Y_test, model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss


SGD_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train)).batch(320)
smlp_SGD = SimpleMLP()
SGD_model = smlp_SGD.build(784, 10)

SGD_model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)

# fit the SGD training data to model
_ = SGD_model.fit(SGD_dataset, epochs=100, verbose=0)

# test the SGD global model and print out metrics
for (X_test, Y_test) in test_batched:
    SGD_acc, SGD_loss = test_model(X_test, Y_test, SGD_model, 1)


def non_iid_x(image_list, label_list, x=1, num_intraclass_clients=10):
    ''' creates x non_IID clients
    args:
        image_list: python list of images or data points
        label_list: python list of labels
        x: none IID severity, 1 means each client will only have one class of data
        num_intraclass_client: number of sub-client to be created from each none IID class,
        e.g for x=1, we could create 10 further clients by splitting each class into 10

    return - dictionary
        keys - clients's name,
        value - client's non iid 1 data shard (as tuple list of images and labels) '''

    non_iid_x_clients = dict()

    # create unique label list and shuffle
    unique_labels = np.unique(np.array(label_list))
    random.shuffle(unique_labels)

    # create sub label lists based on x
    sub_lab_list = [unique_labels[i:i + x] for i in range(0, len(unique_labels), x)]

    for item in sub_lab_list:
        class_data = [(image, label) for (image, label) in zip(image_list, label_list) if label in item]

        # decouple tuple list into seperate image and label lists
        images, labels = zip(*class_data)

        # create formated client initials
        initial = ''
        for lab in item:
            initial = initial + lab + '_'

        # create num_intraclass_clients clients from the class
        intraclass_clients = create_clients(list(images), list(labels), num_intraclass_clients, initial)

        # append intraclass clients to main clients'dict
        non_iid_x_clients.update(intraclass_clients)

    return non_iid_x_clients


pass
