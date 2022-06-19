import tensorflow as tf
import tensorflow.keras as tfk

class ResidualBlock(tf.keras.layers.Layer):
    '''
    '''
    def __init__(self, k, filter_size, kernel_initializer = 'he_normal',kernel_regularizer = None, dropout= 0.5, name='res_block', **kwargs):
        super(ResidualBlock, self).__init__(name=name)
        self.k = k # k filters 
        self.filter_size = filter_size
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.dropout = dropout
        
        self.conv2_a = tfk.layers.Conv2D(self.k, (self.filter_size, self.filter_size), padding='same', kernel_initializer= self.kernel_initializer, kernel_regularizer=self.kernel_regularizer)
        self.bn_a = tfk.layers.BatchNormalization()
        self.do_a = tfk.layers.Dropout(self.dropout)
        self.act_a = tfk.layers.Activation('swish')

        self.conv2_b = tfk.layers.Conv2D(self.k, (self.filter_size, self.filter_size), padding='same', kernel_initializer=self.kernel_initializer ,kernel_regularizer=self.kernel_regularizer )
        self.bn_b = tfk.layers.BatchNormalization()

        self.short = tfk.layers.Conv2D(self.k, (1, 1), padding='same', kernel_initializer=self.kernel_initializer ,kernel_regularizer=self.kernel_regularizer )
        self.short_bn = tfk.layers.BatchNormalization()
        self.short_do = tfk.layers.Dropout(self.dropout)

                
        super(ResidualBlock, self).__init__(**kwargs)

    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({"k": self.k})
        config.update({"filter_size": self.filter_size})
        config.update({"kernel_regularizer": self.kernel_regularizer})
        config.update({"kernel_initializer": self.kernel_initializer})
        config.update({"dropout": self.dropout})
        return config

    def call(self, x):
        conv = self.conv2_a(x)
        conv = self.bn_a(conv)
        conv = self.do_a(conv)
        conv = self.act_a(conv)

        conv = self.conv2_b(conv)
        conv = self.bn_b(conv)

        short = self.short(x)
        short = self.short_bn(short)
        conv = self.short_do(conv)

        res = tfk.layers.add([short,conv])
        res = tfk.layers.Activation('swish')(res)
        return res

    
class UpSamplingAttentionBlock(tf.keras.layers.Layer):
    '''
    Attention is not what you need
    '''
    def __init__(self, k, filter_size, kernel_initializer = 'he_normal',kernel_regularizer = None, name='attention_block', **kwargs):
        super(UpSamplingAttentionBlock, self).__init__(name=name)
        self.k = k # k filters 
        self.filter_size = filter_size
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer        
        
        self.conv2_a = tfk.layers.Conv2D(self.k, (1, 1), padding='same', kernel_initializer=self.kernel_initializer ,kernel_regularizer=self.kernel_regularizer )
        self.conv2_skip = tfk.layers.Conv2D(self.k, (1, 1), strides = (2,2), padding='same', kernel_initializer=self.kernel_initializer ,kernel_regularizer=self.kernel_regularizer )
        self.conv2_c = tfk.layers.Conv2D(1, (1, 1), padding='same', kernel_initializer=self.kernel_initializer ,kernel_regularizer=self.kernel_regularizer )
        
        super(UpSamplingAttentionBlock, self).__init__(**kwargs)

    def get_config(self):
        config = super(UpSamplingAttentionBlock, self).get_config()
        config.update({"k": self.k})
        config.update({"filter_size": self.filter_size})
        config.update({"kernel_regularizer": self.kernel_regularizer})
        config.update({"kernel_initializer": self.kernel_initializer})
        return config

    def call(self, g,skip):
        '''attention for upsamplings'''
        g = self.conv2_a(g)
        x = self.conv2_skip(skip)
        x = tfk.layers.add([g,x])
        x = tfk.layers.Activation('swish')(x)
        
        #x = tfk.layers.UpSampling2D(size = (self.pool_size, self.pool_size))(x) # here 
        x = self.conv2_c(x)
        x = tfk.layers.Activation('sigmoid')(x)
        x = tfk.layers.UpSampling2D(size = (2, 2))(x) # or here
        out = tfk.layers.multiply([x,skip])
        return out, x    
    
    
class UpSamplingBlock(tf.keras.layers.Layer):
    '''
    Attention is not what you need
    '''
    def __init__(self, k, filter_size, kernel_initializer = 'he_normal',kernel_regularizer = None, name='attention_block', **kwargs):
        super(UpSamplingBlock, self).__init__(name=name)
        self.k = k # k filters 
        self.filter_size = filter_size
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer        
        
        self.conv2_a = tfk.layers.Conv2D(n, ( self.filter_size,  self.filter_size), padding='same', kernel_initializer=self.kernel_initializer ,kernel_regularizer=self.kernel_regularizer )
        
        super(UpSamplingBlock, self).__init__(**kwargs)

    def get_config(self):
        config = super(UpSamplingBlock, self).get_config()
        config.update({"k": self.k})
        config.update({"filter_size": self.filter_size})
        config.update({"kernel_regularizer": self.kernel_regularizer})
        config.update({"kernel_initializer": self.kernel_initializer})
        return config

    def call(self, g,skip):
        g = tfk.layers.UpSampling2D(size = (self.pool_size, self.pool_size))(g) # or here
        g = tfk.layers.concatenate([g,skip])
        g = self.conv2_a(g)
        g = tfk.layers.BatchNormalization()(g)
        g = tfk.layers.Activation('swish')(g)
        return g
    
