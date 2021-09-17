from sklearn.ensemble import RandomForestRegressor
import numpy as np
from skimage.filters import gaussian
def background_estimation(image,sigma = 20,grid_separation = 10,offset = 10,n_regressors = 1):
    image = gaussian(image,sigma)
    me = np.mean(image)
    iamge = image-me
    sr,sc = image.shape
    
    rr,cc = np.meshgrid(np.arange(offset,sr-offset,grid_separation),np.arange(offset,sc-offset,grid_separation))    
    rr=rr.ravel()
    cc=cc.ravel()
    b = image[rr,cc]
    idx = np.arange(len(rr))
    
    #result = np.zeros_like(rr_all)*1.0
    N = int(0.5*len(rr)) # 5 % od pixels 
    #for i in range(n_regressors):
    np.random.shuffle(idx)        
    rfc = RandomForestRegressor(n_estimators=50)
    rfc.fit(np.array([rr[idx[:N]],cc[idx[:N]]]).T,b[idx[:N]])
    
    rr_all,cc_all = np.meshgrid(np.arange(sr),np.arange(sc))
    rr_all=rr_all.ravel()
    cc_all=cc_all.ravel()
    
    result = rfc.predict(np.array([rr_all,cc_all]).T)
    result = np.reshape(result/n_regressors,(sc,sr)).T + me
    return result