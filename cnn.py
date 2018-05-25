
'''Convolutional Neural Network based on Keras API for classification
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from os import path, listdir, remove
from dataset_loader import DatasetLoader
import threading
import time

class ConvolutionalNeuralNetwork:
    '''
    SIZE_WIDTH = 78
    SIZE_HEIGHT = 63
    DIMENSION = 1
    input_shape = (SIZE_WIDTH, SIZE_HEIGHT, DIMENSION)
    batch_size = 8
    num_classes = 2
    epochs = 20'''
    thread_flag = 0
    #x_train = x_train.astype('float32')
    #x_test = x_test.astype('float32')
    #x_train /= 255
    #x_test /= 255
    #print('x_train shape:', x_train.shape)
    #print(x_train.shape[0], 'train samples')
    #print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    #y_train = keras.utils.to_categorical(y_train, num_classes)
    #y_test = keras.utils.to_categorical(y_test, num_classes)

    def clear_files(self, path_dir, num_models = 5, delay_thread=5):
        while self.thread_flag != 2:
            files = listdir(path_dir)
            print("files: ", files)
            if len(files) > num_models:
                #print("files: ", files)
                files = sorted(files)
                print("\nfiles_sort: ", files)
                remove_files = files[0:len(files) - num_models]
                print("\nremove_files: ", remove_files)
                self.remove_files(path_dir, remove_files)
                #time.sleep(delay_thread)
            if self.thread_flag == 1:
                self.thread_flag = 2
                break
            time.sleep(delay_thread)

    def remove_files(self, path_dir, remove_files):
        for remove_file in remove_files:
            print ("path: ", path.join(path_dir, remove_file))
            remove(path.join(path_dir, remove_file))

    def build_network (self, input_shape, num_classes):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape = input_shape))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes, activation='softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

    def train (self, images, labels, output_models, name_model, clazzes,
               validation_split, size_image, dimension, batch_size=8, epochs=20):

        input_shape = (size_image[0], size_image[1], dimension)

        dataset = DatasetLoader()
        images, labels = dataset.reshape_base(images, labels, input_shape, len(clazzes))

        self.build_network(input_shape, len(clazzes))

        filepath = path.join(output_models, "weights.{epoch:02d}-{val_loss:.2f}.hdf5")
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        t = threading.Thread(target=self.clear_files, args=(output_models,))
        t.start()

        self.model.fit(images, labels,
                  batch_size = batch_size,
                  callbacks = callbacks_list,
                  epochs = epochs,
                  verbose = 1,
                  #validation_data=(self.data_test, self.labels_test))
                  validation_split = validation_split)

        self.thread_flag = 1
        print("Thread Status: ", t.is_alive())
        while t.is_alive():
            pass

        print("Thread Status: ", t.is_alive())

    def evaluation (self, data_test, labels_test):
        score = model.evaluate(data_test, labels_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
