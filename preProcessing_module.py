import cv2
#import pandas as pd
import numpy as np
#from PIL import Image
import os
#from func_preprocessing import pre_processing
from clazz import Clazz

class PreProcessing:
    def preProcessingAndSegmentation(self, images):
        return images

    def class_to_vec(self, x, num_classes):
        d = np.zeros(num_classes)
        d[x] = 1.0
        print(d)
        return d

    def format_image(self, image, size_image):
        try:
            image = cv2.resize(image, size_image, interpolation = cv2.INTER_CUBIC) / 255.
        except Exception:
            print("[+] Problem during resize")
            return None
        print(image.shape)
        return image

    def initClazzes(self, path):
        dirs = os.listdir(path)
        clazzes = []
        items_by_clazz = []

        for directory in dirs:
            files = os.listdir(os.path.join(path, directory))
            clazz = Clazz(directory, files, len(files))
            clazzes.append(clazz)
            items_by_clazz.append(len(files))

        return clazzes, items_by_clazz, dirs

    def getLabel(self, idClazz, num_classes):
        label = self.class_to_vec(idClazz, num_classes)
        return label

    def getImage(self, path, clazz, img_file, size_image):
        print("[+] filename: ", os.path.join(path, clazz, img_file))
        img = cv2.imread(os.path.join(path, clazz, img_file))
        img = self.preProcessingAndSegmentation(img)
        img = self.format_image(img, size_image)
        return img

    def run(self, path, size_image, dimension):
        images = []
        labels = []
        cont = 0

        clazzes, items_by_clazz, names = self.initClazzes(path)

        for i in range(max(items_by_clazz)):
            idClazz = 0
            for clazz in clazzes:
                if (i < clazz.size_files):
                    print(idClazz)
                    img = self.getImage(path, clazz.name, clazz.files[i], size_image)
                    label = self.getLabel(idClazz, len(clazzes))
                    images.append(img)
                    labels.append(label)
                    cont += 1
                idClazz += 1

        print('\n\n')
        print (cont)
        print (len(images))
        print (len(labels))
        print (names)

        np.save('/tmp/images.npy', images)
        np.save('/tmp/labels.npy', labels)

        images = np.load('/tmp/images.npy')
        labels = np.load('/tmp/labels.npy')

        return images, labels, clazzes
