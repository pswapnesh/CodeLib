"""
Example:
patch_size = 256
scale = 2 #(half the size)
batch_size = 16
lmi = LargeMontageImage(filename, size = patch_size, scale = scale, batch_size = batch_size)
image_patch = lmi.read_window(row_start,col_start,number_of_channels)
y = lmi.map_function(model.predict)
lmi.close()
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pims import ND2_Reader

class LargeMontageImage():
    """
    class to load large images without memory constraints
    """
    def __init__(self,fname, size = 256, scale = 1 , batch_size=16):
        """
        scale is even number for faster processing
        """
        self.fname = fname
        self.img_src = = ND2_Reader(self.fname)
        self.cols = self.img_src.width
        self.rows = self.img_src.height
        
        self.batch_size = batch_size
        self.scale = scale
        self.size = size*self.scale
        
        self.probability_threshold = 0.5
        
        idx_r = np.arange(0,self.rows,self.size)
        if idx_r[-1]+ self.size >self.rows:
            idx_r[-1] = self.rows-self.size
            
        idx_c = np.arange(0,self.cols,self.size)        
        if idx_c[-1]+ self.size >self.cols:
            idx_c[-1] = self.cols-self.size
        
        rr,cc = np.meshgrid(idx_r,idx_c)
        idx_r = rr.ravel()
        idx_c = cc.ravel()
        
        self.idxs = np.vstack((idx_r,idx_c)).T
        
        print("Full image size", self.rows,self.cols)

    def read_window(self,rr,cc,channels):
        size = self.size
        sub_img = np.zeros((size,size,channels))      
        for ii in range(1,channels+1):
            sub_img[:,:,ii-1] = self.img_src.read(ii, window=Window(cc, rr, size, size))
        sub_img/=255.0
        return sub_img[::self.scale,::self.scale,:]
    
    
    def gen_batches(self):
        imgs = []
        idxs = []
        kk = 0
        for ii in range(0,len(self.idxs)):
            img = self.read_window(self.idxs[ii,0],self.idxs[ii,1],3)
            imgs.append(img)
            idxs.append([self.idxs[ii,0],self.idxs[ii,1]])
            kk+=1
            if kk >=self.batch_size:
                kk=0
                yield idxs, np.array(imgs)
                idxs=[]
                imgs=[]
        return 0

    def map_function(self,func):
        """
        apply a image function to the entire image
        """
        y = np.zeros((self.rows,self.cols))
        for idxs,imgs in tqdm(self.gen_batches()):
            yy = func(imgs)
            #yy = resize(yy>self.probability_threshold,(yy.shape[0],self.size,self.size,1))
            yy =  tf.image.resize(yy,[self.size,self.size],antialias=False,method='nearest').numpy()
            for kk,ids in enumerate(idxs):
                y[ids[0]:ids[0]+self.size,ids[1]:ids[1]+self.size] = yy[kk,:,:,0]
        return y
        
    def close(self):
        self.img_src.close()

