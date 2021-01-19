"""
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
encoder_model = ResNet50(input_tensor=inp,weights='imagenet',include_top = False)
for l in encoder_model.layers:
    l.trainable = False
    
unet = UNET(128,3,encoder_model,filter_start=16)
encoder_model = []

"""

from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np

class UNET():
    def __init__(self,size,channels,encoder_model,filter_start = 4,n_classes=1):
        self.encoder_model = encoder_model
        self.size = size
        self.channels = channels
        self.n_classes = n_classes
        self.filter_start = filter_start
        
        self.pool_ker = 2
        self.conv_ker = 3
        self.d_out = 0.25

        self.skips = self.get_skip_connections()
        print('skip connections')
        print(self.skips)
        print('####')

        self.model = self.get_unet()

        self.model.summary()  
        self.encoder_model =[]
        
    def get_skip_connections(self):
        skips = []
        layer_info = []
        layer_shapes = []
        for ii,l in enumerate(self.encoder_model.layers):
            if l.__class__.__name__ == 'Conv2D':
                layer_info.append([l.name,ii,l.output_shape])
                layer_shapes.append(list(l.output_shape)[1])
        layer_shapes = np.array(layer_shapes)
        unique_shapes = np.unique(layer_shapes)
        skips = []
        for sh in unique_shapes:
            idx = np.where(layer_shapes == sh)[0][-1]
            skips.append(layer_info[idx])
        return skips


    def up_net(self,n,c,skip):
        c = tf.keras.layers.UpSampling2D( (self.pool_ker, self.pool_ker) )(c)  
        if skip!= None:         
            c = tf.keras.layers.Concatenate()([c, skip])

        c = tf.keras.layers.Conv2D(n, (self.conv_ker, self.conv_ker), padding='same')(c)
        c = tf.keras.layers.BatchNormalization()(c)
        c = tf.keras.layers.Activation('relu')(c)

        c = tf.keras.layers.Dropout(self.d_out)(c)

        # c = tf.keras.layers.Conv2D(n, (self.conv_ker, self.conv_ker), padding='same')(c)
        # c = tf.keras.layers.BatchNormalization()(c)
        # c = tf.keras.layers.Activation('relu')(c)

        return c   
    
    def get_unet(self):
        # get skip connection layers and its sizes
        skip_layers = []
        for skip_connection in self.skips:
            skip_layers.append(self.encoder_model.layers[skip_connection[1]])

        # make n filters
        filters = 1.0*np.arange(len(skip_layers))*self.filter_start    
        filters=filters[::-1]

        x = skip_layers[0].output
        for ii in range(1,len(skip_layers)):
            x = self.up_net(filters[ii-1],x,skip_layers[ii].output)
        
        x = self.up_net(filters[ii-1],x,None)

        
        outputs = tf.keras.layers.Conv2D(self.n_classes, 1, activation='sigmoid',padding='same',name = 'out')(x)

        unet_model = tf.keras.models.Model(inputs = self.encoder_model.input,outputs = outputs)
        optimizer = 'Adam'
        loss = 'binary_crossentropy'
        metrics = ['mse']
        unet_model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
        return unet_model
    