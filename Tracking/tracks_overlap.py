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
from joblib import Parallel, delayed

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.sparse.csgraph import maximum_bipartite_matching
from skimage.morphology import skeletonize
from skimage.filters import gaussian

# Example usage
"""
from glob import glob
from tracks_overlap_parallel import *
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects

folder = "../inputs/images_pour_swap/"
N = 30
flist = [folder + str(i) + '.tif' for i in range(1,N+1)]
#flist = sorted(glob(folder + '*.tif'))[:10]
def label_reader(ii):
    labeled1 = imread(flist[ii]).astype(int) 
    labeled1 = remove_small_objects(labeled1,256)
    return labeled1

tr = Tracker(flist,label_reader=label_reader,property_names = ['area'],min_overlap_percentage = 5/100,distance_threshold = 100,n_jobs=4)
tr.start()
tr.save('test.csv')
tr.tracks.head()
"""
def center_of_mass(region,intensity):
    skel = skeletonize(region)
    rr,cc = np.where(skel)
    r0,c0 = np.mean(rr),np.mean(cc)
    distances = (rr-r0)**2 + (cc-c0)**2
    idx = np.argmin(distances)
    r0,c0 = rr[idx],cc[idx]
    return r0,c0

def overlap_histogram(region,intensity):
    threshold = 0.02
    tmp = intensity[region]
    uids = np.unique(tmp)          
    s = np.sum(region*1.0)
    hist = np.array([np.sum((tmp==u)*1.0)/s for u in uids])
    if (len(uids)>1) & (uids[0] ==0):
        uids = uids[1:]
        hist = hist[1:]
        
        # idx = np.argmax(hist)
        # uids = [uids[idx]]
        # hist = [hist[idx]]
    if np.max(hist) < threshold:
        return [0],[1.0]
    return uids,hist   



def compute_overlap_cost(props1,props2):
    
    
    labels1 = props1['label'].values
    labels2 = props2['label'].values
    has_overlaps = [False if (len(l)==1) & (l[0]==0) else True for i,(l,h) in enumerate(props2['overlap'])]

    has_overlaps_indices = np.where(has_overlaps)[0]
    cost_matrix = np.zeros((len(has_overlaps_indices),len(labels1)))-np.log(0.01)
    for i,(l,h) in enumerate(props2.loc[has_overlaps_indices,'overlap'].values):
        for l1,h1 in zip(l,h):
            idx = np.where(labels1==l1)[0]
            cost_matrix[i,idx] = -np.log(h1)
    #cost_matrix = cost_matrix- np.min(cost_matrix)         
    return cost_matrix,has_overlaps



def make_tracks(dfs):
    previous_max_id = np.max(dfs[0]['track_id'])

    for i in tqdm(range(1,len(dfs))):
        no_track_ids = dfs[i]['track_id'].values.astype(int) ==-1
        unassigned_ids = np.where(no_track_ids)[0]
        assigned_ids= np.where(np.logical_not(no_track_ids))[0]

        tids = dfs[i].loc[assigned_ids,'track_id'].values.astype(int)
        oids = dfs[i-1]['label'].values.astype(int)
        ids = np.array([np.where(oids == tid)[0] for tid in tids])
        ids = np.squeeze(ids)

        
        trackids = dfs[i-1].loc[ids,'track_id'].values.astype(int)

        dfs[i].loc[assigned_ids,'track_id'] = trackids

        if len(unassigned_ids) > 0:            
            tmp = np.array([previous_max_id + j + 1 for j in range(len(unassigned_ids))])
            dfs[i].loc[unassigned_ids,'track_id'] = tmp
        previous_max_id = max(previous_max_id,np.max(dfs[i]['track_id']))
    return dfs

class Tracker:
    def __init__(self,flist,label_reader,property_names,min_overlap_percentage = 5/100,distance_threshold = 50,n_jobs = 4):
        self.flist = flist
        self.property_names = ['label','bbox'] + property_names
        self.n_jobs = n_jobs
        self.label_reader = label_reader
        self.min_overlap_percentage = min_overlap_percentage
        self.distance_threshold = distance_threshold**2

    def compute_overlaps(self,t):
        l1 = self.label_reader(t-1)
        l2 = self.label_reader(t)
        #properties = pd.DataFrame(regionprops_table(l2,intensity_image = l1,properties=['label','centroid'],extra_properties = (overlap_histogram,)))
        properties = pd.DataFrame(regionprops_table(l2,intensity_image = l1,properties= self.property_names ,extra_properties = (center_of_mass,overlap_histogram)))
        
        properties['center_of_mass-0'] += properties['bbox-0']
        properties['center_of_mass-1'] += properties['bbox-1']

        properties.rename(columns={"center_of_mass-0": "y", "center_of_mass-1": "x", "overlap_histogram": "overlap"}, inplace=True)
        
        tmp_tracks = properties.copy()
        tmp_tracks['label'] = properties['label'].values.astype(int)
        tmp_tracks['t'] = t
        tmp_tracks['distances'] = 11000
        tmp_tracks['track_id'] = -1
        
        return tmp_tracks

    def link_frames(self,df0,df1):

        cost,has_overlaps = compute_overlap_cost(df0,df1)        
        has_overlaps_indices = np.where(has_overlaps)[0]
        
        
        row_ids,col_ids = lsa(cost,maximize = False)
        each_cost = cost[row_ids,col_ids]
        idx = np.where(each_cost < -np.log(self.min_overlap_percentage))[0]      
        df1.loc[has_overlaps_indices[row_ids[idx]],'track_id'] = df0.loc[col_ids[idx],'label'].values.astype(int)
  

        x1 = df1.loc[has_overlaps_indices[row_ids],'x']
        y1 = df1.loc[has_overlaps_indices[row_ids],'y']

        x0 = df0.loc[col_ids,'x']
        y0 = df0.loc[col_ids,'y']

        distances = np.sqrt((x1-x0)**2 + (y1-y0)**2)
        

        df1.loc[has_overlaps_indices[row_ids],'distances'] = distances

        idx_distances = np.where(df1['distances'] > self.distance_threshold)[0]
        df1.loc[idx_distances,'track_id'] = -1


        return df1

    def start(self):
        # first frame
        l1 = self.label_reader(0)
        properties = pd.DataFrame(regionprops_table(l1,intensity_image = l1,properties= self.property_names,extra_properties = (center_of_mass,overlap_histogram,)))
        #print(properties.columns)
        properties['center_of_mass-0'] += properties['bbox-0']
        properties['center_of_mass-1'] += properties['bbox-1']

        properties.rename(columns={"center_of_mass-0": "y", "center_of_mass-1": "x", "overlap_histogram": "overlap"}, inplace=True)
        
        df0 = properties.copy()
        df0['label'] = df0['label'].values.astype(int)
        df0['t'] = 0
        df0['distances'] = 11000
        df0['track_id'] = df0['label'] 

        print('computing overlaps ...')
        dfs = Parallel(n_jobs=self.n_jobs)(delayed(self.compute_overlaps)(t) for t in tqdm(range(1,len(self.flist))))
        #dfs = [self.compute_overlaps(t) for t in tqdm(range(1,len(self.flist)))]
        dfs = [df0] + dfs

        print('linking frames ...')
        dfs = Parallel(n_jobs=self.n_jobs)(delayed(self.link_frames)(dfs[i],dfs[i+1].copy()) for i in tqdm(range(len(dfs)-1)))
        dfs = [df0] + dfs

        print('making tracks ...')
        dfs = make_tracks(dfs)        
        self.tracks = pd.concat(dfs)

        print('computing tracks properties...')
        self.make_velocities()
        return self.tracks


    def save(self,fname):
        self.tracks['track_id'] = self.tracks['track_id'].values.astype(int)
        self.tracks.to_csv(fname,index = None)

    def make_velocities(self):
        sigma = 1.5
        tids = self.tracks['track_id'].unique()
        
        self.tracks['vx'] = 0
        self.tracks['vy'] = 0
        self.tracks['pRev'] = 0
        for tid in tqdm(tids):
            mask = self.tracks['track_id'].values == tid
            idx = np.where(mask)[0]

            vx = np.diff(self.tracks[mask].x)
            vy = np.diff(self.tracks[mask].y)
            vx = gaussian(vx,sigma)
            vy = gaussian(vy,sigma)

            speed = np.sqrt(vx**2 + vy**2)
            speed = gaussian(speed,sigma)

            stoppings = np.array([np.max(o) for i,o in self.tracks[mask].overlap.values])
            stoppings = gaussian(stoppings,1)

            stoppings = np.exp(-0.5e19 * speed/stoppings[1:])

            self.tracks.loc[mask,'vx'] = np.concatenate(([0],vx))
            self.tracks.loc[mask,'vy'] = np.concatenate(([0],vy))            
            self.tracks.loc[mask,'pRev'] = np.concatenate(([0],stoppings))
