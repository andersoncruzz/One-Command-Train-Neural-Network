from os.path import join
import numpy as np
import cv2

class DatasetLoader(object):

    def __init__(self):
        pass

    def load_from_file(self, path_dir):
        images = np.load(path_dir + "data.npy")
        labels = np.load(path_dir + "label.npy")
        print("[+] Load database from file")
        return images, labels

    def reshape_base(self, images, labels, input_shape, num_classes):
        print("[+] Database reshape")
        images = images.reshape([-1, input_shape[0], input_shape[1], input_shape[2]])
        labels = labels.reshape([-1, num_classes])
        return images, labels

