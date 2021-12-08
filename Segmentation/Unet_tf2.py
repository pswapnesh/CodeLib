import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import io
#from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from IPython import display
import tensorflow as tf
import sys, time
from scipy.stats import entropy
tf.config.run_functions_eagerly(True)
print(tf.__version__)

class UNET():

    def __init__(self,img_size = 256, channels = 1,cnet = [8,16,32],nclasses = 4, fname = 'unet_patches.hdf5', log_dir = '../logs/'):

        self.img_rows = img_size
        self.img_cols = img_size
        self.channels = channels
        self.n_classes = nclasses
        
        self.cnet = cnet
        self.upnet = cnet[:-1][::-1];
        self.pool_ker = 2
        self.conv_ker = 9
        self.d_out = 0.5
        
        self.modelSavedName = fname;
        
        self.log_dir = log_dir
        self.summary_writer = tf.summary.create_file_writer(log_dir + 'loss/')
        self.image_writer = tf.summary.create_file_writer(log_dir + 'imgs/')
        
        self.kernel_constraints =tfk.constraints.MaxNorm(max_value=1)# tfk.constraints.NonNeg() # 
        
        initial_learning_rate = 0.05
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)
        
        self.opt = tfk.optimizers.Adam(learning_rate=self.lr_schedule)
        
        self.model = self.get_unet()
                
        self.model.summary()       
        

    def down_net(self,n,c1,first=False):
        conv_ker = self.conv_ker;
        pool_ker = self.pool_ker;
        if not first:
            c1 = tfk.layers.MaxPooling2D((pool_ker, pool_ker)) (c1)
        #  kernel_constraint = self.kernel_constraints,  
        c1 = tfk.layers.Conv2D(n, (conv_ker, conv_ker), activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer='l1_l2') (c1)
        c1 = tfk.layers.BatchNormalization()(c1)
        c1 = tfk.layers.Dropout(self.d_out) (c1)
        #c1 = Conv2D(n, (conv_ker, conv_ker), activation='relu', padding='same') (c1)
        #c1 = BatchNormalization()(c1)
        return c1
      
    def up_net(self,n,c4,c6):
        conv_ker = self.conv_ker;
        pool_ker = self.pool_ker;
        
        u6 = tfk.layers.UpSampling2D(size = (pool_ker, pool_ker)) (c6)                
        u6 = tfk.layers.concatenate([u6, c4],axis = 3)        
        c6 = tfk.layers.Conv2D(n, (conv_ker, conv_ker), activation='relu', padding='same', kernel_initializer='he_normal',kernel_regularizer='l1_l2') (u6)
        c6 = tfk.layers.BatchNormalization()(c6)
        c6 = tfk.layers.Dropout(self.d_out) (c6)
        #c6 = Conv2D(n, (conv_ker, conv_ker), activation='relu', padding='same', kernel_initializer='he_normal') (c6)
        #c6 = BatchNormalization()(c6)
        return c6  
       

    def get_unet(self):
        _epsilon = tf.convert_to_tensor(K.epsilon(), np.float32)
        inp = tfk.layers.Input(shape = (self.img_rows, self.img_cols, self.channels))                
        targets = tfk.layers.Input((self.img_rows, self.img_cols, 1))
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
          
        outputs = tfk.layers.Conv2D(self.n_classes, 1, activation='sigmoid', kernel_initializer='he_normal',padding='same')(conv)        
        model = tfk.models.Model(inputs=inp, outputs=outputs)        
        return model
    
    def loss_function(self,y_true,y_pred,d):
        loss_val = K.binary_crossentropy(y_true,y_pred)
        #loss_val = tfk.losses.CategoricalCrossentropy(from_logits=True)(y_true,y_pred)
#         weighted_loss_val = ( 1 + 0.5*d) * loss_val
        weighted_loss_val = (1+ 2*K.exp(d)) * loss_val
        return  weighted_loss_val - K.log(self.jaccard_coef(y_true,y_pred)) #- tf.keras.metrics.AUC(thresholds=[0.7,0.8,0.9,0.95])(y_true,y_pred)  #+ 0.1*self.get_weights_entropy()
    
    @tf.function    
    def step(self, X,y,d):
        # keep track of our gradients
        with tf.GradientTape() as tape:
            # make a prediction using the model and then calculate the
            # loss
            ypred = self.model(X, training=True)
            loss = self.loss_function(y,ypred,d)
            # calculate the gradients using our tape and then update the
            # model weights
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return tfk.metrics.Mean('train_loss', dtype=tf.float32)(loss)
    
    def test_step(model, x_test, y_test):
        ypred = self.model(x_test)
        loss = self.loss_function(y,ypred)
        return tfk.metrics.Mean('train_loss', dtype=tf.float32)(loss)


    def train(self,training_set,num_epochs = 20,steps_per_epoch = 20,save_folder = None,display_results = False): 
        template = 'Epoch {}, Loss: {}'
        # display        
#         for t in training_set.take(1):         
#             xtest,ytest = t[0][0:1,:,:,:],t[1][0:1,:,:,:]

        xtest = imread('../inputs/full_data/defocused_misic/test.tif') 
        xtest = rescale(normalize2max(xtest),2)
        ytest = imread('../inputs/full_data/defocused_misic/testy.tif') 
        # epochs
        for i in range(0,num_epochs):            
            display.clear_output(wait=True)
            self.display(xtest,ytest,i,save_folder,display_results)
            epoch_losses = []
            k = 0
            for x_batch, y_batch, d_batch in training_set:
                loss = self.step(x_batch,y_batch,d_batch)
                epoch_losses.append(loss)            
                if k > steps_per_epoch:
                    break;
                k+=1
            mean_loss = np.mean(np.array(epoch_losses))
            print(template.format(i+1,mean_loss))
            with self.summary_writer.as_default():
                tf.summary.scalar('loss', mean_loss, step=i)
            
                
    def get_weights_entropy(self):
        weights = self.model.layers[1].get_weights()
        w = weights[0]
        tmp = w[:,:,0,:].reshape(7*7,w.shape[-1])
        me = np.mean(tmp,axis = 1)
        s = np.sum(np.array([entropy(tmp[:,i],me) for i in range(tmp.shape[1])]))
        if np.isfinite(s):
            return s
        return 20
            
    def display(self,xx,yy,epoch,save_folder,display_results):
        
#         plt.clf()
        
        weights = self.model.layers[1].get_weights()
        w = weights[0]
        
        n = w.shape[-1]
        fig, axs = plt.subplots(8,int(n/8),figsize=(int(n/8)/2,4))
        axs = axs.ravel()        
        for i in range(len(axs)):
            axs[i].imshow(w[:,:,0,i],cmap = 'gray')
            axs[i].set_axis_off()
        plt.tight_layout(h_pad = 0.5,w_pad = 0.5)
        if save_folder is not None:
            #plt.savefig(save_folder + str(epoch) + '_wts.png', bbox_inches='tight',pad_inches = 0)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight',pad_inches = 0)
            plt.close(fig)
            buf.seek(0)
            im = tf.image.decode_png(buf.getvalue(), channels=4)
            im = tf.expand_dims(im, 0)
            with self.image_writer.as_default():
                tf.summary.image("Weights0", im, step=epoch) 
            
        if display_results:
            plt.show()
        
#         weights = self.model.layers[5].get_weights()
#         w = weights[0]
#         n = w.shape[-1]
        
#         fig, axs = plt.subplots(4,int(n/4),figsize=(int(n/8),2))
#         axs = axs.ravel()        
#         for i in range(len(axs)):
#             axs[i].imshow(w[:,:,0,i],cmap = 'gray')
#             axs[i].set_axis_off()
#         plt.tight_layout(h_pad = 0.5,w_pad = 0.5)
#         if save_folder is not None:
#             plt.savefig(save_folder + str(epoch) + '_wts9.png', bbox_inches='tight',pad_inches = 0)
#             buf = io.BytesIO()
#             plt.savefig(buf, format='png', bbox_inches='tight',pad_inches = 0)
#             buf.seek(0)
#             im = imread(buf)
#             with self.image_writer.as_default():
#                 tf.summary.image("test data", im[np.newaxis,:,:,:], step=epoch)            
#             #im = imread(buf)
#             #im.show()
#             buf.close()
            
            
#         if display_results:
#             plt.show()
        
        
        yp = self.model.predict(-shapeindex_preprocess(xx)[np.newaxis,:256,:256,:])
        
        fig, ax = plt.subplots(1,3,figsize = (8,2))
        ax[0].imshow(xx[:256,:256],cmap = 'gray')        
        ax[0].set_axis_off()
        ax[1].imshow(yy[:256,:256],cmap = 'gray')    
        ax[1].set_axis_off()
        ax[2].imshow(yp[0,:,:,0],cmap = 'gray')           
        ax[2].set_axis_off()        
        plt.tight_layout()
        if save_folder is not None:
            #plt.savefig(save_folder + str(epoch) + '.png', bbox_inches='tight',pad_inches = 0)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight',pad_inches = 0)
            plt.close(fig)
            buf.seek(0)
            im = tf.image.decode_png(buf.getvalue(), channels=4)
            im = tf.expand_dims(im, 0)
            with self.image_writer.as_default():
                tf.summary.image("test data", im, step=epoch)   
            
        if display_results:
            plt.show()        
        plt.clf()
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
