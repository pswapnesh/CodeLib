import tensorflow as tf
import tensoflow.keras.backend as K

def dice_coef(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def jaccard_coef(y_true, y_pred):
    smooth = K.epsilon()
    #y_pred = K.cast(K.greater(y_pred, .8), dtype='float32') # .5 is the threshold
    #y_true = K.cast(K.greater(y_true, .9), dtype='float32') # .5 is the threshold
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)    

def pixel_weighted_cross_entropy(targets,weights, predictions):
    loss_val = tf.keras.losses.binary_crossentropy(targets, predictions)
    weighted_loss_val = weights[:,:,:,0] * loss_val
    return K.mean(weighted_loss_val) - K.log(jaccard_coef(targets,predictions))

def bce_and_jac(y_true,y_pred):
    return tf.keras.losses.binary_crossentropy(y_true,y_pred)-K.log(jaccard_coef(y_true,y_pred))    


#### object wise jaccard    