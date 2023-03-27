# Author: [Swapnesh panigrahi @ team IAM.LCB]

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from skimage.io import imread,imsave
from skimage.measure import label,regionprops,regionprops_table
from skimage.morphology import remove_small_objects
from skimage.segmentation import find_boundaries
from sklearn.neighbors import KDTree
from skimage import metrics
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import linear_sum_assignment as lsa

from skimage.color import label2rgb
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from IPython.display import clear_output

# Example usage
"""
from glob import glob
from tracks_overlap import *

folder = "../inputs/segmented_cropped_ws/"
flist = sorted(glob(folder + '*.tif'))
def label_reader(ii):
    yy = imread(flist[ii])/255.0    
    yy = remove_small_objects(yy>0.9,256)
    labeled1 = label(yy)
    return labeled1

tr = Tracker(flist,label_reader=label_reader)
tr.start()
tr.save('outputs/tracks.csv')
"""

def read_labeled_image(f):
    """
    Reads an image file from the given file path and returns the labeled image.

    Parameters:
    f (str): File path of the input image.

    Returns:
    labeled1 (numpy.ndarray): Labeled image.
    """
    yy = imread(f)/255.0    
    yy = remove_small_objects(yy>0.9,256)
    labeled1 = label(yy)
    return labeled1

def relabel(labeled_image,oldlabels,newlabels):
    """
    Given a labeled image, relabels the image with new label values, based on the old labels and new labels.

    Parameters:
    labeled_image (numpy.ndarray): Labeled image.
    oldlabels (list): List of old label values.
    newlabels (list): List of new label values corresponding to old label values.

    Returns:
    newimage (numpy.ndarray): Relabeled image.
    """
    newimage = np.zeros_like(labeled_image)
    for old_label,new_label in zip(oldlabels,newlabels):
        newimage[labeled_image==old_label] = new_label
    return newimage

def get_max_label(p):
    """
    Given a regionprops object, returns the maximum intensity value and its corresponding label value in the region.

    Parameters:
    p (skimage.measure._regionprops.RegionProperties): Regionprops object.

    Returns:
    uids (numpy.ndarray): Array of label values.
    hist (list): List of intensity values corresponding to the label values.
    """    
    tmp = p.intensity_image[p.image]
    uids = np.unique(tmp)          
    hist = [np.sum((tmp==u)*1.0) for u in uids]    
    return uids,hist

def overlap_method(labeled1,labeled2):
    """
    Given two labeled images, returns a cost matrix that indicates the cost of assigning each region in labeled2 to each region in labeled1.

    Parameters:
    labeled1 (numpy.ndarray): Labeled image 1.
    labeled2 (numpy.ndarray): Labeled image 2.

    Returns:
    cost_matrix (numpy.ndarray): Cost matrix.
    """
    uids1 = np.unique(labeled1)[1:]
    overlap = np.logical_and(labeled1 >0,labeled2 >0)
    overlap = (overlap*1.0)*labeled1
    props = regionprops(labeled2,intensity_image = overlap)
    cost_matrix = np.zeros((len(props),len(uids1)))
    for i in range(len(props)):
        labels,histo = get_max_label(props[i]) 
        if (len(labels) == 1 ) and (labels[0] == 0):
            cost_matrix = np.concatenate((cost_matrix,np.zeros_like(cost_matrix[:,0:1])),axis = 1)
            cost_matrix[i,-1] = histo[0]
            continue    
        for j,h in zip(labels[1:],histo[1:]):            
            j = int(j-1)
            cost_matrix[i,j] = h
    return cost_matrix


class Tracker:
    """
    Initializes the Tracker object.

    Parameters:
    flist (list): List of file paths.
    label_reader (function): Function for reading labeled images.

    Attributes:
    flist (list): List of file paths.
    label_reader (function): Function for reading labeled images.
    labeled1 (numpy array): Labeled image of the first frame.
    uids1 (list): List of unique labels in labeled1.
    tracks (pandas DataFrame): DataFrame containing track information.
    previous_max (int): Maximum label used so far for tracks.
    """
    def __init__(self,flist,label_reader):
        self.flist = flist
        self.label_reader = label_reader
        ii = 0
        #self.labeled1 = read_labeled_image(self.flist[ii])
        self.labeled1 = self.label_reader(ii)
        self.uids1 = np.unique(self.labeled1)[1:]

        self.tracks = self.temp_track(self.labeled1,0)
        self.tracks['track_ids'] = self.tracks['original_labels']
        
        self.previous_max = np.max(self.tracks['track_ids'].values)

    def temp_track(self,labeled,tt):
        properties = pd.DataFrame(regionprops_table(labeled,properties = ['centroid','label']))
        tmp_tracks = pd.DataFrame(columns = ['t','y','x','original_labels','track_ids'])
        tmp_tracks['y'] = properties['centroid-0']
        tmp_tracks['x'] = properties['centroid-1']
        tmp_tracks['original_labels'] = properties['label']
        tmp_tracks['t'] = tt
        return tmp_tracks

    def clean_null(self):        
        idx = self.tracks['track_ids'].isnull()
        self.tracks.loc[idx,'track_ids'] = [self.previous_max + i + 1 for i in range(len(np.where(idx)[0]))]
        self.previous_max = np.max(self.tracks['track_ids'].values)

    

    def start(self,dt = 1,max_distance = 15):
        for ii in tqdm(range(dt,len(self.flist),dt)):
            #labeled2 = read_labeled_image(self.flist[ii])
            labeled2 = self.label_reader(ii)
            uids2 = np.unique(labeled2)[1:]

            tmp_tracks = self.temp_track(labeled2,ii)

            cost_matrix = overlap_method(self.labeled1,labeled2)
            row_ids,col_ids = lsa(cost_matrix,maximize = True)
            idx = self.tracks['t']== ii-dt
            
            #tmp_tracks.loc[row_ids,'track_ids'] = tracks.loc[idx,:].loc[col_ids,'original_labels']
            for r,c in zip(row_ids,col_ids):                   
                if c >= len(self.uids1):
                    tmp_tracks.loc[r,'track_ids'] = self.previous_max
                    self.previous_max+=1       
                else:
                    dx = (tmp_tracks.loc[r,'x'] - self.tracks.loc[idx,:].loc[c,'x'])**2 
                    dy = (tmp_tracks.loc[r,'y'] - self.tracks.loc[idx,:].loc[c,'y'])**2 
                    d = np.sqrt(dx+dy)
                    if d > max_distance:
                        tmp_tracks.loc[r,'track_ids'] = self.previous_max
                        self.previous_max+=1    
                    else:
                        tmp_tracks.loc[r,'track_ids'] = self.tracks.loc[idx,:].loc[c,'track_ids']
                
                    #assigned += [tracks.loc[idx,:].loc[c,'original_labels'].value]
            
            self.labeled1 = np.copy(labeled2)
            self.uids1 = np.unique(self.labeled1)[1:]

            self.tracks = pd.concat((self.tracks,tmp_tracks))
            #self.clean_null()
        print('total tracks detected: ', np.max(self.tracks['track_ids']))

    def save(self,fname):
        self.tracks.to_csv(fname)
