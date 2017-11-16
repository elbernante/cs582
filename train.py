
# coding: utf-8

# In[1]:


import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from lib.stats_tools import dice_coef, dice_coef_loss, f1_score
from lib.callbacks import TrainMonitor
from lib.util import image_augment, get_train_val_sets

from config import *

K.set_image_data_format('channels_last')  # TF dimension ordering in this code


# In[2]:


def make_model(input_shape, learning_rate):
    inputs = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
#     print("Output shape:", conv10._keras_shape)

    model = Model(inputs=[inputs], outputs=[conv10])
    
    model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef, f1_score])
#     model.compile(optimizer=Adam(lr=learning_rate), loss='cosine', metrics=[dice_coef, f1_score])
    
    return model


# In[3]:


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


# In[4]:


def predict(X, batch_size=8):
    model = load_model(WEIGHTS_FILE, verbose=1,
                       custom_objects={'dice_coef_loss': dice_coef_loss, 
                                       'dice_coef': dice_coef,
                                       'f1_score': f1_score})
    return model.predict(X, batch_size=batch_size)

