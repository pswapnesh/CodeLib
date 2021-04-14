import tensorflow as tf
from skimage.io import imread,imsave
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from IPython import display
import numpy as np

## General utilities
def plotter(images,cmap = 'jet',size = (15,15)):
    N = len((images))
    if N <=3:
        n1,n2 = 1,N
    else:
        n1 = int(np.ceil(np.sqrt(N)))
        n2 = n1
    fig,axes = plt.subplots(n1,n2,figsize = size)
    ax = axes.ravel()
    for ii,im in enumerate(images):
        ax[ii].imshow(im,cmap)
        plt.tight_layout()
    plt.show()



class DataLoader():

    def __init__(self,x_list,y_list,test_size = 0.2 ,preprocess = lambda x,y:(x,y)):
        seed = 42
        self.x_list = x_list
        self.y_list = y_list            
        self.test_size = test_size
        self.preprocess = preprocess        
        print("length of x,y: ", len(x_list),',',len(x_list))
        xx = imread(self.x_list[0])
        yy = imread(self.y_list[0])
        xx,yy = self.preprocess(xx,yy)
        if len(xx.shape)==2:
            self.channels = 1
            self.rows,self.cols = xx.shape
        else:
            self.rows,self.cols,self.channels = xx.shape
        if len(yy.shape)==2:
            self.classes = 1
        else:
            _,_,self.classes = yy.shape
        
        print(self.rows,self.cols,self.channels,self.classes)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.x_list, self.y_list, test_size=self.test_size, random_state=seed)

        self.training_dataset = self.get_tf_dataset('train')    
        self.test_dataset = self.get_tf_dataset('test')    
        
    
    def train_generator(self):
        """ generator that generates one train record of X and y randomly from the file lists"""
        ii = np.random.randint(len(self.X_train))
        xx = imread(self.X_train[ii])
        yy = imread(self.y_train[ii])
        xx,yy = self.preprocess(xx,yy)        
        yield xx,yy

    def test_generator(self):
        """ generator that generates one test record of X and y randomly from the file lists"""
        ii = np.random.randint(len(self.X_test))
        xx = imread(self.X_test[ii])
        yy = imread(self.y_test[ii])
        xx,yy = self.preprocess(xx,yy)  
        yield xx,yy

    def get_tf_dataset(self,dset = 'train' ,BATCH_SIZE=16):
        if dset == 'train':
            dataset = tf.data.Dataset.from_generator(self.train_generator,
                                                    (tf.float32, tf.float32),
                                                    ((self.rows,self.cols,self.channels), (self.rows,self.cols,self.classes)))
        else:
            dataset = tf.data.Dataset.from_generator(self.test_generator,
                                                (tf.float32, tf.float32),
                                                ((self.rows,self.cols,self.channels), (self.rows,self.cols,self.classes)))
        BUFFER_SIZE = 2*BATCH_SIZE
        train_dataset = dataset.repeat().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        return train_dataset

    def show(self):
        for t in self.training_dataset.take(1):
            plotter([t[0][0,:,:,0],t[1][0,:,:,0]],'gray')

            
class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self,dl):
        self.dl =dl
    
    def on_epoch_begin(self, epoch, logs=None):
        display.clear_output(wait=True)
        for t in self.dl.training_dataset.take(1):
            xx,yy = t[0][0:1,:,:,:],t[1][0:1,:,:,:]
            yp = self.model.predict(xx)
            plt.clf()
            fig, ax = plt.subplots(1,3)
            ax[0].imshow(xx[0,:,:,0])        
            ax[1].imshow(yy[0,:,:,0])            
            ax[2].imshow(yp[0,:,:,0])   
            plt.axis('off')
            plt.show()