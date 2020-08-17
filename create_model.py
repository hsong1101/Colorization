#!/usr/bin/env python
# coding: utf-8

# In[80]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import time

import tensorflow as tf
from tensorflow.keras import layers

import warnings
warnings.filterwarnings('ignore')

# This suppress future warning of tensorflow
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Without these lines, bert will error out with : Blas GEMM launch failed
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


# In[81]:


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


# In[82]:


def get_image(file_path):
    
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = img / 255
    
    return img


def train_model(epochs, optimizer, train_dataset, valid_dataset, loss_fn, loss_metric, 
                save_path='models', 
                batch_size=16,
                save_iter_num=10, 
                ver_iter_num=10):
    
    
    # Loss values for plotting
    train_loss_values = []
    
    epoch = 1
    
    prev_loss = 0 
    spent_time = 0
    
    steps = tf.data.experimental.cardinality(train_ds).numpy() // batch_size
    
    start = time.time()
    
    while epoch < (epochs+1):
        
        # Iterate over the batches of the dataset.
        for step, rgb_images in enumerate(train_dataset):

            with tf.GradientTape() as tape:

                # Convert to Images
                # Create grayscale images for y
                gray_images = tf.image.rgb_to_grayscale(rgb_images)

                pred_images = model(gray_images)

                loss = loss_fn(rgb_images, pred_images)

                loss += sum(model.losses)
                

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_metric(loss)

            loss_result = loss_metric.result()

            train_loss_values.append(loss_result)

            if step >= steps:
                break

                
        if epoch % ver_iter_num == 0:

            print(f'Epoch : {epoch}, Loss : {loss_result}, It took {round(time.time() - start, 3)} secs')
            
            with open('plots/losses.txt', 'a') as file:
                file.writelines(str(loss_result.numpy()) + '\n')
        
            start = time.time()

            
        if epoch % save_iter_num == 0:
            
            total_loss = 0
            
            for rgb_images in valid_dataset:
                
                gray = tf.image.rgb_to_grayscale(rgb_images)
                pred = model(gray)
                loss = loss_fn(rgb_images, pred)
                loss += sum(model.losses)
                
                loss_metric(loss)
                loss_result = loss_metric.result()
                total_loss += loss_result
                
            with open('plots/valid_losses.txt', 'a') as file:
                file.writelines(str(total_loss.numpy()) + '\n')
                
            
            print('\nSaving model\n')
            model.save_weights(f'{save_path}/model.ckpt')
            
            
        if epoch % 1000 == 0:
            
            try:
                for i in valid_dataset.take(1):
                    img = i[:1]

                    pred = tf.image.rgb_to_grayscale(img)
                    pred = model(pred)
                    pred = pred.numpy()

                fig, ax = plt.subplots(1, 3, figsize=(20, 12))

                ax[0].imshow(cv2.cvtColor(img.numpy()[0], cv2.COLOR_RGB2GRAY), cmap='gray')
                ax[1].imshow(pred[0])
                ax[2].imshow(img.numpy()[0])

                ax[0].set_title('Grayscale')
                ax[1].set_title('Pred RGB')
                ax[2].set_title('True RGB')

                fig.savefig(f'plots/Model_Epoch{epoch}.png')
            except:
                pass
            
        epoch += 1
            
    return train_loss_values


# In[71]:


dataset = tf.data.Dataset.list_files('data/images/*.jpg')

train_ds = dataset.skip(200)
val_ds = dataset.take(200)

batch_size = 16

train_ds = train_ds.map(get_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size).shuffle(1)
valid_ds = val_ds.map(get_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)


# In[ ]:


model = ColorModel((224,224, 3), 3, 2)

loss_fn = tf.keras.losses.MeanSquaredError()
# optimizer = tf.optimizers.SGD(learning_rate=1e-7, momentum=.3, decay=.99)
optimizer= tf.optimizers.Adam(1e-4, amsgrad=True)
loss_metric = tf.keras.metrics.Mean()

h = train_model(50000, optimizer, train_ds, valid_ds, loss_fn, loss_metric, batch_size=batch_size, ver_iter_num=1, save_iter_num=50)


# In[118]:


# with open('plots/losses.txt', 'r') as file:
#     loss = file.readlines()
#     loss = pd.Series(loss)
#     loss = loss.str.replace("\n", '')
#     loss = loss.astype('float32')
    
# with open('plots/valid_losses.txt', 'r') as file:
#     val_loss = file.readlines()
#     val_loss = pd.Series(val_loss)
#     val_loss = val_loss.str.replace("\n", '')
#     val_loss = val_loss.astype('float32')
    
# plt.figure(figsize=(8, 8))

# plt.plot(loss[-1000:], label='Train Loss')
# # plt.plot(val_loss, label='Valid Loss')

# plt.legend()
# plt.show()


# In[120]:


# z = ColorModel((224, 224, 3),3,2)
# z.load_weights('models/model.ckpt')

# for i in train_ds.take(1):
#     img = i[:1]
    
#     pred = tf.image.rgb_to_grayscale(img)
#     pred = z(pred)
#     pred = pred.numpy()
    
# fig, ax = plt.subplots(1, 3, figsize=(20, 12))

# ax[0].imshow(cv2.cvtColor(img.numpy()[0], cv2.COLOR_RGB2GRAY), cmap='gray')
# ax[1].imshow(pred[0])
# ax[2].imshow(img.numpy()[0])

# ax[0].set_title('Grayscale')
# ax[1].set_title('Pred RGB')
# ax[2].set_title('True RGB')

# plt.show()

# # fig.savefig('plots/Model3_Epoch3000.png')


# In[ ]:




