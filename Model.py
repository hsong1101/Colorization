import tensorflow as tf
from tensorflow.keras import layers


class ResidualBlock(tf.keras.layers.Layer):
    
    def __init__(self,  in_channel, out_channel, kernel_size=3, strides=2, dropout=.3, use_pooling=False, activation='relu'):
        
        super(ResidualBlock, self).__init__()
        
        self.use_pooling = use_pooling
        
        self.conv1 = layers.Conv2D(in_channel, kernel_size, strides=1, padding='same', activation=activation, dtype='float32')
        self.conv2 = layers.Conv2D(in_channel, kernel_size, strides=1, padding='same', activation=activation, dtype='float32')
        self.conv3 = layers.Conv2D(out_channel, kernel_size, strides=strides, padding='same', activation=activation, dtype='float32')
        
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        
        self.dropout = layers.Dropout(dropout)
        
    def call(self, x):
        
        output = self.dropout(self.conv1(x))
        output = self.dropout(self.conv2(output))
        output += x

        if self.use_pooling:
            output = layers.MaxPooling2D()(output)
        else:
            output = self.conv3(output)
            output = self.bn1(output)
        
        return output

    
class ColorModel(tf.keras.Model):
    
    def __init__(self, input_shape, kernel_size=3, strides=2, dropout=.3, use_pooling=False):
        
        super(ColorModel, self).__init__()
        
        self.res1 = ResidualBlock(3, 16, kernel_size, strides, use_pooling=use_pooling)
        self.res2 = ResidualBlock(16, 32, kernel_size, strides, use_pooling=use_pooling)
        self.res3 = ResidualBlock(32, 64, kernel_size, strides, use_pooling=use_pooling)
        self.res4 = ResidualBlock(64, 128, kernel_size, strides, use_pooling=use_pooling)
        self.res5 = ResidualBlock(128, 256, kernel_size, strides, use_pooling=use_pooling)
        self.res6 = ResidualBlock(256, 512, kernel_size, strides, use_pooling=use_pooling)
        self.res7 = ResidualBlock(512, 1024, kernel_size, strides, use_pooling=use_pooling)
        
        # No reducing dimension
        self.seq1 = ResidualBlock(64, 256, kernel_size, strides=1)
        self.seq2 = ResidualBlock(256, 256, kernel_size, strides=1)
        self.seq3 = ResidualBlock(256, 256, kernel_size, strides=1)
        self.seq4 = ResidualBlock(256, 256, kernel_size, strides=1)
        self.seq5 = ResidualBlock(256, 256, kernel_size, strides=1)
        
        # Dense
        self.dense1 = layers.Dense(1024, activation='relu')
        self.dense2 = layers.Dense(512, activation='relu')
        self.dense3 = layers.Dense(256, activation='relu')
        
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()
        
        self.dropout = layers.Dropout(dropout)
        
        # Reduce Dimensionality
        self.out1 = ResidualBlock(512, 256, strides=1)
        self.out2 = ResidualBlock(256, 128, strides=1)
        self.out3 = ResidualBlock(128, 2, strides=1, activation='sigmoid')
        
        self.shape = input_shape
        
        
        
    def call(self, x):
        
        out = self.res1(x)
        out = self.res2(out)
        out = self.res3(out)
        
        # Run Sequential Processing
        seq = self.seq1(out)
        seq = self.seq2(seq)
        seq = self.seq3(seq)
        seq = self.seq4(seq)
        seq = self.seq5(seq)

        # Fully Connected
        flat = self.res4(out)
        flat = self.res5(flat)
        flat = self.res6(flat)
        flat = self.res7(flat)
        flat = layers.Flatten()(flat)
        
        flat = self.dropout(self.bn1(self.dense1(flat)))
        flat = self.dropout(self.bn2(self.dense2(flat)))
        flat = self.dropout(self.bn3(self.dense3(flat)))

        flat = tf.expand_dims(flat, axis=1)
        flat = tf.repeat(flat, 28, axis=1)
        flat = tf.expand_dims(flat, axis=2)
        flat = tf.repeat(flat, 28, axis=2)

        merged = tf.concat([seq, flat], axis=-1)
    
        # Upsample images
        out = self.out1(merged)
        out = tf.image.resize(out, [56, 56])
        
        out = self.out2(out)
        out = tf.image.resize(out, [112, 112])
        
        out = self.out3(out)
        out = tf.image.resize(out, [224, 224])

        out = tf.concat([x, out], axis=-1)
        
        return out
    
    
    def summary(self):
        
        inputs = layers.Input(shape=self.shape)
        
        return tf.keras.Model(inputs=[inputs], outputs=self.call(inputs)).summary()