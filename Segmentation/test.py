import tensorflow as tf 
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

from Unet_from_encoder import *

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

size = 128
inp = tf.keras.layers.Input(shape = (size,size,3))
encoder_model = MobileNetV2(input_tensor=inp,weights='imagenet',include_top = False)
for l in encoder_model.layers:
    l.trainable = False

unet = UNET(128,3,encoder_model,filter_start=16)
encoder_model = []
