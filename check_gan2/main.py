'''
DCGAN on MNIST using Keras
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''

import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self, sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time))
# end


def discriminator(img_rows, img_cols, channel):
    D = Sequential(name="discriminator")
    depth = 64
    dropout = 0.4
    # In: 28 x 28 x 1, depth = 1
    # Out: 14 x 14 x 1, depth=64
    input_shape = (img_rows, img_cols, channel)
    D.add(Conv2D(depth * 1, 5, strides=2, input_shape=input_shape, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))

    D.add(Conv2D(depth * 2, 5, strides=2, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))

    D.add(Conv2D(depth * 4, 5, strides=2, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))

    D.add(Conv2D(depth * 8, 5, strides=1, padding='same'))
    D.add(LeakyReLU(alpha=0.2))
    D.add(Dropout(dropout))

    # Out: 1-dim probability
    D.add(Flatten())
    D.add(Dense(1))
    D.add(Activation('sigmoid'))
    D.summary()
    return D
# end


def generator(img_dim):
    G = Sequential(name="generator")
    dropout = 0.4
    depth = 64 + 64 + 64 + 64
    dim = img_dim//4     # 7
    # In: 100
    # Out: dim x dim x depth
    G.add(Dense(dim * dim * depth, input_dim=100))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))
    G.add(Reshape((dim, dim, depth)))
    G.add(Dropout(dropout))

    # In: dim x dim x depth
    # Out: 2*dim x 2*dim x depth/2
    G.add(UpSampling2D())
    G.add(Conv2DTranspose(int(depth / 2), 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    G.add(UpSampling2D())
    G.add(Conv2DTranspose(int(depth / 4), 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    G.add(Conv2DTranspose(int(depth / 8), 5, padding='same'))
    G.add(BatchNormalization(momentum=0.9))
    G.add(Activation('relu'))

    # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
    G.add(Conv2DTranspose(1, 5, padding='same'))
    G.add(Activation('sigmoid'))
    G.summary()
    return G
# end


class DCGAN(object):
    def __init__(self, img_rows, img_cols, channel):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        else:
            self.D = discriminator(self.img_rows, self.img_cols, self.channel)
            return self.D

    def generator(self):
        if self.G:
            return self.G
        else:
            self.G = generator(self.img_rows)
            return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return self.AM
# end


class MNIST_DCGAN(object):
    def __init__(self):
        # self.img_rows = 28
        # self.img_cols = 28
        # self.channel = 1
        #
        # self.x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
        # self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, 1).astype(np.float32)

        # self.load_data_mnist()
        self.load_data_gray()
        # self.load_data_color()

        self.DCGAN = DCGAN(self.img_rows, self.img_cols, self.channel)
        self.discriminator = self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()
    # end

    def load_data_mnist(self):
        self.x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1
        self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, self.channel).astype(np.float32)
        self.img_name = "mnist_{}.png"

    def load_data_gray(self):
        from PIL import Image
        from path import Path as path
        from numpy import asarray
        imgs = []
        for img_file in path("E:/Datasets/CelebA/gray_32").files("*.jpg"):
            img = Image.open(img_file)
            imgs.append(asarray(img))
            # if len(imgs) >= 60000: break

        self.x_train = np.array(imgs)
        self.img_rows = 32
        self.img_cols = 32
        self.channel = 1
        self.x_train = (self.x_train.astype(np.float32) / 255)
        self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, self.channel)
        self.img_name = "gray_{}.png"

    def load_data_color(self):
        from PIL import Image
        from path import Path as path
        from numpy import asarray
        imgs = []
        for img_file in path("E:/Datasets/CelebA/color_32").files("*.jpg"):
            img = Image.open(img_file)
            imgs.append(asarray(img))
            # if len(imgs) >= 60000: break

        self.x_train = np.array(imgs)
        self.img_rows = 32
        self.img_cols = 32
        self.channel = 3
        self.x_train = (self.x_train.astype(np.float32) / 255)
        self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, self.channel)
        self.img_name = "color_{}.png"

    def train(self, train_steps=2000, batch_size=256, save_interval=0, verbose=False):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)

            if verbose or save_interval > 0 and (i == 0 or (i+1)%save_interval == 0):
                log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
                log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
                print(log_mesg)

            if save_interval > 0:
                if i == 0 or (i+1)%save_interval == 0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0], noise=noise_input, step=(i+1))
    # end

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = self.img_name.format(0)
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = self.img_name.format(step)
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()
    # end
# end


if __name__ == '__main__':
    mnist_dcgan = MNIST_DCGAN()
    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=20000, batch_size=256, save_interval=500)
    timer.elapsed_time()
    mnist_dcgan.plot_images(fake=True)
    mnist_dcgan.plot_images(fake=False, save2file=True)