import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,Reshape,concatenate,Conv2DTranspose,UpSampling2D,Multiply,Lambda,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
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

    def __init__(self,img_size = 256, channels = 1,cnet = [8,16,32],nclasses = 4, fname = 'unet_patches.hdf5', log_dir = './logs/'):

        self.img_rows = img_size
        self.img_cols = img_size
        self.channels = channels
        self.n_classes = nclasses
        self.cnet = cnet
        self.upnet = cnet[:-1][::-1];
        self.pool_ker = 2
        self.conv_ker = 3
        self.d_out = 0.5
        #K.set_image_dim_ordering('tf')
        self.modelSavedName = fname;
        
        self.log_dir = log_dir
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        
        self.model = self.get_unet()
        self.opt = Adam()
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
        _epsilon = tf.convert_to_tensor(K.epsilon(), np.float32)
        inp = Input(shape = (self.img_rows, self.img_cols, self.channels))        
        weight_inp = Input((self.img_rows, self.img_cols, 1))
        targets = Input((self.img_rows, self.img_cols, 1))
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
          
        
        #outputs = Conv2D(self.n_classes , (1, 1), activation='sigmoid',padding='same') (conv)
        outputs = Conv2D(self.n_classes, 1, activation='sigmoid', kernel_initializer='he_normal',padding='same')(conv)        
        model = Model(inputs=inp, outputs=outputs)        
        return model
    
    def loss_function(self,y_true,y_pred,wts):
        return K.binary_crossentropy(y_true,y_pred)
    
    @tf.function    
    def step(self, X,y):
        # keep track of our gradients
        with tf.GradientTape() as tape:
            # make a prediction using the model and then calculate the
            # loss
            ypred = self.model(X, training=True)
            loss = self.custom_loss(y,ypred)
            # calculate the gradients using our tape and then update the
            # model weights
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    # train for pixelweighted                      
    def train(self,train_x,train_y,wts,n_epochs=5,batch_sz=10):       
        numUpdates = int(train_x.shape[0] / batch_sz)
        # loop over the number of epochs
        for epoch in range(0, n_epochs):
            display.clear_output(wait=True)
            self.generate_and_save_images(epoch + 1,train_x[:4,:,:,:])
    
            # show the current epoch number
            print("[INFO] starting epoch {}/{}...".format(epoch + 1, n_epochs), end="")
            sys.stdout.flush()
            epochStart = time.time()
            # loop over the data in batch size increments
            for i in range(0, numUpdates):
                # determine starting and ending slice indexes for the current
                # batch
                start = i * batch_sz
                end = start + batch_sz
                # take a step                
                loss = self.step(train_x[start:end],wts[start:end], train_y[start:end])         
            with self.summary_writer.as_default():
                tf.summary.scalar('loss', loss, step=epoch)
            # show timing information for the epoch
            epochEnd = time.time()
            elapsed = (epochEnd - epochStart) / 60.0
            print("took {:.4} minutes".format(elapsed))
            
    # train for pixelweighted
    def fit(self,train_ds, epochs, test_ds):
        for epoch in range(epochs):
            start = time.time()

            display.clear_output(wait=True)

            for example_input, example_target in test_ds.take(1):
                self.generate_images( example_input, example_target)
            print("Epoch: ", epoch)

            # Train
            for n, (input_image, target) in enumerate(train_ds.take(16)):
                print('.', end='')
                if (n+1) % 100 == 0:
                    print()
                self.step(input_image, target)
                if n > 32:
                    break;
            print()

#             # saving (checkpoint) the model every 20 epochs
#             if (epoch + 1) % 20 == 0:
#                 checkpoint.save(file_prefix = checkpoint_prefix)

            print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                              time.time()-start))
        #checkpoint.save(file_prefix = checkpoint_prefix)
                      
    def pixel_weighted_cross_entropy(self,targets,weights, predictions):
        loss_val = tf.keras.losses.binary_crossentropy(targets, predictions)
        weighted_loss_val = weights[:,:,:,0] * loss_val
        return K.mean(weighted_loss_val) - K.log(self.jaccard_coef(targets,predictions)) 

    def custom_loss(self,y_true,y_pred):
        return  K.binary_crossentropy(y_true,y_pred) - K.log(self.jaccard_coef(y_true,y_pred)) 
        #return  K.binary_crossentropy(y_true,y_pred) - MeanIoU(num_classes=2)(y_true,y_pred)
        
    def jaccard_coef(self,y_true, y_pred):
        smooth = K.epsilon()
        #y_pred = K.cast(K.greater(y_pred, .8), dtype='float32') # .5 is the threshold
        #y_true = K.cast(K.greater(y_true, .9), dtype='float32') # .5 is the threshold
        intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
        sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

        jac = (intersection + smooth) / (sum_ - intersection + smooth)

        return K.mean(jac)
    
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
    def generate_images(self,test_input, tar):
        prediction = self.model(test_input, training=False)
        plt.figure(figsize=(15,20))

        display_list = [test_input[0], tar[0,:,:,0],tar[0,:,:,1],prediction[0,:,:,0],prediction[0,:,:,1]]
        title = ['Input Image', 'Ground Truth','Ground Truth','Myxo', 'Coli']

        for i in range(5):
            plt.subplot(1, 5, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()