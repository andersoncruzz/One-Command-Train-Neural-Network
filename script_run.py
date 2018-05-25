from cnn import ConvolutionalNeuralNetwork
from preProcessing_module import PreProcessing

PATH = '/home/anderson/gitoyota/poc_toyota/tire_model_recognition/CV/base_train'
SIZE_IMAGE = (223, 143)
DIMENSION = 3

pre = PreProcessing()
images, labels, clazzes = pre.run(path = PATH, size_image = SIZE_IMAGE, dimension = DIMENSION)

OUTPUT_MODELS = "/home/anderson/gitoyota/poc_toyota/keras_automatic/models_keras"
NAME_MODEL = "test"
VALIDATION_SPLIT = 0.75


net = ConvolutionalNeuralNetwork()
net.train(images=images, labels=labels, output_models = OUTPUT_MODELS, name_model = NAME_MODEL, clazzes = clazzes,
          validation_split = VALIDATION_SPLIT, size_image = SIZE_IMAGE, dimension = DIMENSION)
