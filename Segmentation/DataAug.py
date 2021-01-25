import tensorflow as tf
import numpy as np

class Augumentation():
    def __init__(self,size = 512):
        self.seed = 42
        self.size = size
        self.transform_functions = [self.crop,
                                   self.rotate]

    def transform(self,xx,yy):
        """ choose a random transfrom to be applied to both X and y"""
        func = np.random.choice(self.transform_functions)
        return func(xx,yy)                  

    def rotate(self,xx,yy):
        factor = 3.1416/8.0
        seed = np.random.randint(10000)
        rotate = tf.keras.layers.experimental.preprocessing.RandomRotation(factor, fill_mode='nearest', interpolation='bilinear', seed=seed)
        return rotate(xx),rotate(yy)

    def crop(self,xx,yy):
        seed = np.random.randint(10000)
        height_factor = 0.9
        width_factor = 0.9
        crop = tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor, width_factor, fill_mode='reflect',interpolation='bilinear', seed=seed)
        return crop(xx),crop(yy)



# from skimage.filters import gaussian
# from skimage.util import random_noise
# from skimage.exposure import adjust_gamma
# from skimage.transform import resize,rotate





# class Augumentation():
#     def __init__(self,size = 512, seed = 42):
#         self.seed = seed
#         np.random.seed(self.seed)
#         self.size = size
#         self.transform_functions = [self.crop,
#                                    self.blur,
#                                    self.additive_noise,
#                                    self.blur_and_noise,
#                                    self.poisson_noise,
#                                    self.contrast,
#                                    self.rotate]
        
#     def transform(self,xx,yy):
#         func = np.random.choice(self.transform_functions)
#         return func(xx,yy)
    
#     def crop(self,xx,yy):
#         r0,r1 = np.random.randint(64),-1-np.random.randint(64)
#         c0,c1 = np.random.randint(64),-1-np.random.randint(64)
#         xx= resize(xx[r0:r1,c0:c1],(self.size,self.size),preserve_range = True)
#         yy = resize(yy[r0:r1,c0:c1,:],(self.size,self.size),preserve_range = True)  
#         return xx,yy
    
#     def blur(self,xx,yy):
#         return gaussian(xx,1.5*np.random.rand()),yy
    
#     def blur_and_noise(self,xx,yy):
#         xx,yy = self.blur(xx,yy)
#         return self.additive_noise(xx,yy)
    
#     def additive_noise(self,xx,yy):
#         return random_noise(xx,mode = 'gaussian',var = 0.1*np.random.rand()),yy
    
#     def poisson_noise(self,xx,yy):
#         return random_noise(xx,mode = 'poisson'),yy
    
#     def contrast(self,xx,yy):
#         return adjust_gamma(xx,0.25 + np.random.rand()),yy
    
#     def rotate(self,xx,yy):
#         ang = np.random.randint(2*20)-20        
#         xx= rotate(xx[r0:r1,c0:c1],ang,preserve_range = True)
#         yy = resize(yy[r0:r1,c0:c1,:],ang,preserve_range = True) 
#         return xx,yy
