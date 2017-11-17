
# coding: utf-8

# In[1]:


import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Concatenate, Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from lib.stats_tools import dice_coef, dice_coef_loss, f1_score, f2_score, f05_score
from lib.callbacks import TrainMonitor
from lib.util import image_augment, get_train_val_sets

from config import *

K.set_image_data_format('channels_last')  # TF dimension ordering in this code


# In[2]:


def Inception_B(filter_size, activation='relu', padding='valid'):
    def _layer(input_tensor):
        f1 = filter_size // 4
        tower_1 = Conv2D(f1, (1, 1), activation=activation, padding=padding)(input_tensor)
        tower_1 = Conv2D(f1, (1, 7), activation=activation, padding=padding)(tower_1)
        tower_1 = Conv2D(f1, (7, 1), activation=activation, padding=padding)(tower_1)
        tower_1 = Conv2D(f1, (1, 7), activation=activation, padding=padding)(tower_1)
        tower_1 = Conv2D(f1, (7, 1), activation=activation, padding=padding)(tower_1)
        
        tower_2 = Conv2D(f1, (1, 1), activation=activation, padding=padding)(input_tensor)
        tower_2 = Conv2D(f1, (1, 7), activation=activation, padding=padding)(tower_2)
        tower_2 = Conv2D(f1, (7, 1), activation=activation, padding=padding)(tower_2)
        
        tower_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
        tower_3 = Conv2D(f1, (1, 1), activation=activation, padding=padding)(tower_3)
        
        tower_4 = Conv2D(f1, (1, 1), activation=activation, padding=padding)(input_tensor)
        
        return Concatenate(axis=-1)([tower_1, tower_2, tower_3, tower_4])
    return _layer

def Inception_C(filter_size, activation='relu', padding='valid'):
    def _layer(input_tensor):
        f1, f2 = filter_size // 4, filter_size // 8
        
        tower_1 = Conv2D(f2, (1, 1), activation=activation, padding=padding)(input_tensor)
        tower_1 = Conv2D(f2, (3, 3), activation=activation, padding=padding)(tower_1)
        tower_11 = Conv2D(f2, (1, 3), activation=activation, padding=padding)(tower_1)
        tower_12 = Conv2D(f2, (3, 1), activation=activation, padding=padding)(tower_1)
        
        tower_2 = Conv2D(f2, (1, 1), activation=activation, padding=padding)(input_tensor)
        tower_21 = Conv2D(f2, (1, 3), activation=activation, padding=padding)(tower_2)
        tower_22 = Conv2D(f2, (3, 1), activation=activation, padding=padding)(tower_2)
        
        tower_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input_tensor)
        tower_3 = Conv2D(f1, (1, 1), activation=activation, padding=padding)(tower_3)
        
        tower_4 = Conv2D(f1, (1, 1), activation=activation, padding=padding)(input_tensor)
        
        return Concatenate(axis=-1)([tower_11, tower_12, tower_21, tower_22, tower_3, tower_4])
        
    return _layer

def Reduction(filter_size, strides=(2, 2), activation='relu'):
    def _layer(input_tensor):
        tower_1 = Conv2D(filter_size, (1, 1), activation=activation, padding='same')(input_tensor)
        tower_1 = Conv2D(filter_size, (3, 3), activation=activation, padding='same')(tower_1)
        tower_1 = Conv2D(filter_size, (3, 3), strides=strides, activation=activation, padding='same')(tower_1)
        
        tower_2 = Conv2D(filter_size, (1, 1), activation=activation, padding='same')(input_tensor)
        tower_2 = Conv2D(filter_size, (3, 3), strides=strides, activation=activation, padding='same')(tower_2)
        
        tower_3 = MaxPooling2D(pool_size=strides)(input_tensor)
        
        return Concatenate(axis=-1)([tower_1, tower_2, tower_3])
    return _layer



# In[3]:


def make_model(input_shape, learning_rate):
    inputs = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation='selu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='selu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    print("Pool 1:", pool1._keras_shape)

#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
#     conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2 = Inception_B(64, activation='selu', padding='same')(pool1)
    print("Conv 2:", conv2._keras_shape)
    pool2 = Reduction(16, (2, 2), activation='selu')(conv2)
    print("Pool 2:", pool2._keras_shape)

#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
#     conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv3 = Inception_B(128, activation='selu', padding='same')(pool2)
    print("Conv 3:", conv3._keras_shape)
    pool3 = Reduction(32, (2, 2))(conv3)
    print("Pool 3:", pool3._keras_shape)

#     conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
#     conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv4 = Inception_B(256, activation='selu', padding='same')(pool3)
    pool4 = Reduction(64, (2, 2))(conv4)
    print("Pool 4:", pool4._keras_shape)

#     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
#     conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = Inception_C(512, activation='selu', padding='same')(pool4)
    print("Conv 5:", conv5._keras_shape)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
#     conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
#     conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = Inception_B(256, activation='selu', padding='same')(up6)
    print("Conv 6:", conv6._keras_shape)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
#     conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
#     conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = Inception_B(128, activation='selu', padding='same')(up7)
    print("Conv 7:", conv7._keras_shape)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
#     conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
#     conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)
    conv8 = Inception_B(64, activation='selu', padding='same')(up8)
    print("Conv 8:", conv8._keras_shape)
    
    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='selu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='selu', padding='same')(conv9)
    print("Conv 9:", conv9._keras_shape)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    print("Output shape:", conv10._keras_shape)

    model = Model(inputs=[inputs], outputs=[conv10])
    
    model.compile(optimizer=Adam(lr=learning_rate, decay=0.),
                  loss=dice_coef_loss,
                  metrics=[dice_coef, f1_score, f2_score, f05_score])
#     model.compile(optimizer=Adam(lr=learning_rate), loss='cosine', metrics=[dice_coef, f1_score])
    
    return model


# In[4]:


model = make_model((IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS), LEARNING_RATE)


# In[5]:


model.count_params()


# ## U-NET Model
# Parameter count: 7,759,521
# 
# Pool 1: (None, 48, 64, 32)
# 
# Pool 2: (None, 24, 32, 64)
# 
# Pool 3: (None, 12, 16, 128)
# 
# Pool 4: (None, 6, 8, 256)
# 
# Conv 5: (None, 6, 8, 512)
# 
# Conv 6: (None, 12, 16, 256)
# 
# Conv 8: (None, 48, 64, 64)
# 
# Conv 8: (None, 48, 64, 64)
# 
# Output shape: (None, 96, 128, 1)

# In[6]:


def train():
    X_train, y_train, X_val, y_val = get_train_val_sets(TRAIN_SET_PICKLE, VALIDATION_SET_PICKLE)
#     X_train, y_train, X_val, y_val = X_train[:16,...], y_train[:16,...], X_val[:16,...], y_val[:16,...]
    model = make_model((IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS), LEARNING_RATE)
    model_checkpoint = ModelCheckpoint(WEIGHTS_FILE, monitor='val_loss', save_best_only=True)
    train_monitor = TrainMonitor(HISTORY_LOG, X_train[:1,...], y_train[:1,...], out_dir="output/preds/")
    
    return model.fit_generator(image_augment(X_train, y_train, batch_size=TRAIN_BATCH_SIZE, seed=10), 
                        y_train.shape[0] // TRAIN_BATCH_SIZE,
                        epochs=EPOCHS_TO_RUN,
                        verbose=1,
                        callbacks=[model_checkpoint, train_monitor], 
                        validation_data=(X_val, y_val))
hist = train()


# In[7]:


def predict(X, batch_size=8):
    model = load_model(WEIGHTS_FILE, verbose=1,
                       custom_objects={'dice_coef_loss': dice_coef_loss, 
                                       'dice_coef': dice_coef,
                                       'f1_score': f1_score})
    return model.predict(X, batch_size=batch_size)

