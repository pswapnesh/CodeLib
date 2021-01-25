import matplotlib.pyplot as plt
import numpy as np

def plotter(images,grid = None,cmap = 'jet',size = (15,15),axis = 'off'):
    """ grid plot an array of images 
    >>> plotter([im0,im1,im2,im3])
    """
    if grid==None:

        N = len((images))
        if N <=5:
            n1,n2 = 1,N
        else:
            prefer = [3,4,5]
            idx = np.argmin(N%prefer)
            n2 = prefer[idx]
            n1 = int(np.ceil(N/n2))
    else:
        n1,n2 = grid    

    fig,axes = plt.subplots(n1,n2,figsize = size)
    ax = axes.ravel()
    for ii,im in enumerate(images):
        ax[ii].imshow(im,cmap)
        plt.axis(axis)
    plt.tight_layout()
    plt.show()

