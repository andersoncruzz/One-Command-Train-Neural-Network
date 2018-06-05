from cnn import ConvolutionalNeuralNetwork
from preProcessing_module import PreProcessing

PATH = '/home/neural/projetos/One-Command-Train-Neural-Network/images'
SIZE_IMAGE = (75, 75)
DIMENSION = 3

pre = PreProcessing()
images, labels, clazzes = pre.run(path = PATH, size_image = SIZE_IMAGE, dimension = DIMENSION)

OUTPUT_MODELS = "/home/neural/projetos/One-Command-Train-Neural-Network/output_models"
NAME_MODEL = "test"
VALIDATION_SPLIT = 0.2


net = ConvolutionalNeuralNetwork()
net.train(images=images, labels=labels, output_models = OUTPUT_MODELS, name_model = NAME_MODEL, clazzes = clazzes,
          validation_split = VALIDATION_SPLIT, size_image = SIZE_IMAGE, dimension = DIMENSION, batch_size=4, epochs = 100)
