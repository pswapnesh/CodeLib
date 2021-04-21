from scipy.ndimage import label
from skimage.io import imread,imsave
from skimage.measure import regionprops
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# replace labels with its property
def label_comprehend(labeled,props,measurement = 'area'):
    #props = regionprops(labeled)
    res = np.zeros_like(labeled)*1.0
    for p in props:
        msk = labeled == p.label
        res[msk] = p[measurement]
    return res

# get local iou for each pixel in the union
def get_local_iou_image(yt,yp,verbose = False):
  
  global_union = np.logical_or(yt,yp)
  global_intersection = np.logical_and(yt,yp)

  label_inter,count_inter = label(global_intersection)
  label_union,count_union = label(global_union)
  
  props_inter = regionprops(label_inter)
  props_union = regionprops(label_union)


  local_union = label_comprehend(label_union,props_union,'area')
  local_inter = label_comprehend(label_inter,props_inter,'area')
  iou_local = local_inter/local_union
  if verbose:
    plt.figure(figsize=(10,10))
    plt.subplot(1,3,1)
    plt.imshow(local_union)
    plt.title('lcal_union')
    plt.subplot(1,3,2)
    plt.imshow(local_inter)
    plt.title('lcal_inter')
    plt.subplot(1,3,3)
    plt.imshow(iou_local)
    plt.title('iou')
    plt.show()

  
  #iou_local[local_union < 1] = -1
  return iou_local

# get the iou inside each cell for both the true iamge and the predicted image
def get_cell_iou(yt,yp,iou_local):

  label_true,count_true = label(yt)
  label_pred,count_pred = label(yp)

  props_true = regionprops(label_true,intensity_image = iou_local)
  props_pred = regionprops(label_pred,intensity_image = iou_local)

  
  iou_true = np.array([p.max_intensity for p in props_true])
  iou_pred = np.array([p.max_intensity for p in props_pred])
  
  return iou_true,iou_pred

# get confusions matrix of cell detection based on IOU threshold
def cell_wise_confusion(iou_true,iou_pred,iou_thresholds = np.arange(0.000,1,0.001)):
    true_truth = np.ones_like(iou_true) > 0 # all cells are true cells
    pred_truth = iou_pred > 0.001 # all cells with intersection > 0 are true cells
    fpr = []
    tpr = []
    fnr = []
    for t in iou_thresholds:
      cm_t = confusion_matrix(true_truth,iou_true > t)/len(iou_true)  
      cm_p = confusion_matrix(pred_truth,iou_pred < t)/len(iou_pred) 
      #true negatives is c00, false negatives is c10, true positives is  c11 and false positives is c01.
      #try:
      if len(cm_t) <2:
        tp,fn = cm_t[0][0],0
      else:
        tp,fn = cm_t[1][1],cm_t[1][0]
        #tn, fp, fn, tp = cm_t.ravel() #tn, fp, fn, tp
      if len(cm_p) <2:
        fp,tn = cm_p[0][0],0
      else:
        fp,tn = cm_p[1,1],cm_p[0,1]
      fnr.append(fn)
      tpr.append(tp)
      fpr.append(fp)

    fnr = np.array(fnr)
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    return iou_thresholds,fpr,tpr,fnr

    
# main function that uses the above function to return metrics like jacard
def get_metrics(yt,yp):
  iou_local = get_local_iou_image(yt,yp)
  iou_true,iou_pred = get_cell_iou(yt,yp,iou_local)
  iou_thresholds,fpr,tpr,fnr = cell_wise_confusion(iou_true,iou_pred,iou_thresholds = np.arange(0.000,1,0.01))
  
  jac = tpr/(tpr+fpr+fnr)
  return iou_thresholds,jac
