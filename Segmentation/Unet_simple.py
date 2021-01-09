"""
Example
#let x_list and y_list be the list of file paths to the dataset where you have images in the format X = (row,cols,channels), y = (rows,cols,classes)
folder = '..\imagefolder\\'
x_list = glob.glob(folder + '*x.tif')
y_list = glob.glob(folder + '*y.tif')
len(x_list),len(y_list)

##
dl = DataLoader(x_list,y_list)
#show data
dl.show()

#make model
unet = UNET(dl.rows,dl.channels,dl.classes)
#define a realtime display callback
disp = DisplayCallback(dl)
#train
unet.model.fit(dl.training_dataset,epochs=10,steps_per_epoch=5,callbacks=[disp])
"""


import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,Reshape,concatenate,Conv2DTranspose,UpSampling2D,Multiply,Lambda,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard,EarlyStopping
from tensorflow.keras.optimizers import Adam,Adagrad,Adadelta
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import MeanIoU,Precision
from time import time 
import matplotlib.pyplot as plt
from IPython import display
import tensorflow as tf
import sys, time
print(tf.__version__)

class UNET():

    def __init__(self,img_size = 256, channels = 1,nclasses = 4,cnet = [16,32,64,128], fname = 'unet_model.h5'):

        self.img_rows = img_size
        self.img_cols = img_size
        self.channels = channels
        self.n_classes = nclasses
        
        self.cnet = cnet
        self.upnet = cnet[:-1][::-1]
        
        self.pool_ker = 2
        self.conv_ker = 3
        self.d_out = 0.5
        
        self.modelSavedName = fname
                
        self.model = self.get_unet()
        self.model.compile(loss=self.custom_loss , optimizer= Adam() , metrics=[self.jaccard_coef, Precision()] )        
        self.model.summary()       

    def down_net(self,n,c1,first=False):
        conv_ker = self.conv_ker;
        pool_ker = self.pool_ker;
        if not first:
            c1 = MaxPooling2D((pool_ker, pool_ker)) (c1)
          
        c1 = Conv2D(n, (conv_ker, conv_ker), activation='relu', padding='same') (c1)
        c1 = BatchNormalization()(c1)
        c1 = Dropout(self.d_out) (c1)
        c1 = Conv2D(n, (conv_ker, conv_ker), activation='relu', padding='same') (c1)
        c1 = BatchNormalization()(c1)
        return c1
      
    def up_net(self,n,c4,c6):
        conv_ker = self.conv_ker;
        pool_ker = self.pool_ker;
        
        u6 = UpSampling2D(size = (pool_ker, pool_ker)) (c6)        
        u6 = concatenate([u6, c4],axis = 3)
        c6 = Conv2D(n, (conv_ker, conv_ker), activation='relu', padding='same', kernel_initializer='he_normal') (u6)
        c6 = BatchNormalization()(c6)
        c6 = Dropout(self.d_out) (c6)
        c6 = Conv2D(n, (conv_ker, conv_ker), activation='relu', padding='same', kernel_initializer='he_normal') (c6)
        c6 = BatchNormalization()(c6)
        return c6  
       

    def get_unet(self):
        """ make the encoder decoder unet model """
        _epsilon = tf.convert_to_tensor(K.epsilon(), np.float32)
        inp = Input(shape = (self.img_rows, self.img_cols, self.channels))                
        convs = []

        for it,cc in enumerate(self.cnet):
            if it==0:
                conv = self.down_net(cc,inp,True)
            else:
                conv = self.down_net(cc,conv)
            convs.append(conv)
        
        print(convs)
        
        for it1,uu in enumerate(self.upnet):
            print(uu,it1,self.upnet,convs[-it1-2],conv)
            conv = self.up_net(uu,convs[-it1-2],conv)
          
        
        outputs = Conv2D(self.n_classes, 1, activation='sigmoid', kernel_initializer='he_normal',padding='same')(conv)        
        model = Model(inputs=inp, outputs=outputs)        
        return model
    

    def custom_loss(self,y_true,y_pred):
        return  K.binary_crossentropy(y_true,y_pred) - K.log(self.jaccard_coef(y_true,y_pred)) 
        
    def jaccard_coef(self,y_true, y_pred):
        smooth = K.epsilon()
        intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
        sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return K.mean(jac)

    def train(self,train_set,val_set,epochs = 10, steps_per_epoch = 16,validation_steps = 2):
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        #disp = DisplayCallback()
                
        self.model.fit(train_set,epochs = 10 ,steps_per_epoch = steps_per_epoch, callbacks = [es,disp])
    
    def generate_and_save_images(self,epoch, test_input):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = self.model(test_input, training=False)

        fig = plt.figure(figsize=(10,10))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        
        plt.show()

    def generate_images(self,training_dataset):
        display.clear_output(wait=True)
        t = training_dataset.take(1)
        xx = t[0][:,:,:,:]
        yy = t[1][:,:,:,:]
            
        yp = self.model(xx, training=False)
        plt.figure(figsize=(15,20))
        display_list = [xx[0,:,:,0],yy[0,:,:,0],yp[0,:,:,0]]
        title = ['Input Image', 'Ground Truth','Prediction']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()
