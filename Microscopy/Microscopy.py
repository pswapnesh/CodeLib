# from Microscopy import Microscopy
# from skimage.morphology import binary_opening
# from skimage.io import imread,imsave
# from skimage.exposure import rescale_intensity
# from skimage.filters import gaussian

from scipy.signal import fftconvolve,unit_impulse
from skimage.filters import gaussian
from scipy.special import jv,airy
import numpy as np
from scipy.signal import convolve2d
class Microscopy():
    def __init__(self):
        self.condensers = {"Ph1": (0.45, 3.75, 24),
                            "Ph2": (0.8, 5.0, 24),
                            "Ph3": (1.0, 9.5, 24),
                            "Ph4": (1.5, 14.0, 24),
                            "PhF": (1.5, 19.0, 25)} #W, R, Diameter
    def psf_pc(self,radius,F,W,R):        
        Lambda = 0.5
        xx,yy = np.meshgrid(np.linspace(-radius,radius,2*radius+1), np.linspace(-radius,radius,2*radius+1))
        scale2 = 10
        xx = xx/scale2
        yy = yy/scale2
        rr = np.sqrt(xx**2 + yy**2)
        x = rr*(2*np.pi)*(1/F)*(1/Lambda)
        #ker = o_airy(rr_dl,R,W)
        ker = R*jv(1,2*3.1416*R*x)/x - (R-W)*jv(1,2*3.1416*(R-W)*x)/x
        ker[radius,radius] = np.nanmax(ker)
        ker = ker/np.linalg.norm(ker)
        ker -= unit_impulse(xx.shape,(radius,radius))    
        ker = gaussian(ker,1)        
        #ker = self.normalize2max(ker)
        ker = ker/np.sum(ker)
        return ker
    # PSFs
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3372640/
    def phase_contrast(self,im,radius=10,condenser='Ph1',F=60):        
        sr,sc = im.shape    
        off = 1
        if sr%2 == 0:
            off = 0
        rr,cc = np.meshgrid(np.arange(-int(sr/2),off+int(sr/2)),np.arange(-int(sc/2),off+int(sc/2)))                  
        scale = 9000
        W,R,D = self.condensers[condenser]
        NA = 0.9
        F = 1.33*D/(2*NA)
        W = W*scale
        R = R*scale
        F = F*scale
        ker = self.psf_pc(radius,F,W,R)
        return fftconvolve(im,ker,'same')#/np.sum(ker)
    
    #https://www.sciencedirect.com/science/article/pii/S0006349517309840
    def bright_field(self,im,z = 0.1,params = [0.01,1.39,0.0005,0.87]):       
        k,n,rho,na = params
        sr,sc = 10,10        
        x,y = np.meshgrid(np.arange(-int(sr/2),int(sr/2)),np.arange(-int(sc/2),int(sc/2)))
        ker = jv(0, k*na*rho*np.sqrt(x**2 + y**2)) * np.sin(k*n*z* (np.sqrt(1 - (na*rho/n)**2)-1))         
        #return fftconvolve(im,fker,'same')
        return convolve2d(im,ker,'same')
    
    def gaussian(self,im,sigma):
        return gaussian(im,sigma)
    
    # Noise in images
    def photon_noise(self,im,photons=10000,ratio = 1.0453 ):
        # ratio :  between 'background' (no cells) and cell wall
        im = (photons*(ratio-1))*im + photons
        return np.random.poisson(im, size=im.shape)
    
    # readout gaussian noise
    def detector_noise(self,im, noise=1):
        return im + np.random.normal(scale=noise, size=im.shape)
    
    def normalize2max(self,im):
        im = im-np.min(im)
        return im/np.max(im)
    

