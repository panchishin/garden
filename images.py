import json
import topwords
import tensorflow as tf
import numpy as np
import os
import urllib
import sys

BASE_URL = 'https://mygardenorg.s3.amazonaws.com/plantifier/'
SIZE = 128


def nameToURL(name):
    return BASE_URL + name


class image_reader_graph:

    def __init__(self, size=SIZE):
        self.img_name = tf.placeholder(dtype=tf.string)
        self.img = tf.image.decode_jpeg(tf.read_file(self.img_name), channels=3)
        self.img = tf.image.resize_image_with_crop_or_pad(self.img, size * 2, size * 2)
        self.img = tf.image.resize_images(self.img, [size, size])
        self.img = self.img / 256.0
        self.img_flip = tf.image.flip_left_right(self.img)


def download(image_name):
    print "downloading", image_name
    urllib.urlretrieve(nameToURL(image_name), 'data/' + image_name)


class from_names:

    def __init__(self, size=SIZE):
        self.names = []
        self.images = []
        self.labels = []
        self.size = size

    def load(self, image_names, label=0):
        reader = image_reader_graph(self.size)
        with tf.Session() as sess:
            count = 0
            for name in image_names:
                count += 1
                if not os.path.isfile("data/" + name):
                    download(name)
                if count % 10 == 0:
                    print "{0:4.1f}%".format(100.0 * count / len(image_names)),
                try:
                    images = sess.run([reader.img, reader.img_flip], feed_dict={reader.img_name: "data/" + name})
                    for image in images:
                        self.names.append(name)
                        self.images.append(image)
                        self.labels.append(label)

                except:
                    print "*** Error on image", name, "***",

            print "{0:4.1f}%".format(100.0)

        self.names = np.array(self.names)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        return self

    def concat(self, another):
        self.names = np.concatenate((self.names, another.names), 0)
        self.images = np.concatenate((self.images, another.images), 0)
        self.labels = np.concatenate((self.labels, another.labels), 0)
        return self

    def slice(self, start=None, end=None):
        skip = 1 if end == None else 2
        start = None if start == None else 2 * start
        end = None if end == None else 2 * end
        result = from_names()
        result.names = self.names[start:end:skip]
        result.images = self.images[start:end:skip]
        result.labels = self.labels[start:end:skip]
        return result

    def shuffle(self):
        arrangement = np.arange(self.labels.shape[0])
        np.random.shuffle(arrangement)
        self.names = self.names[arrangement]
        self.images = self.images[arrangement]
        self.labels = self.labels[arrangement]
