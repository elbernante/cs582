
import os, pickle

from keras.models import Model, load_model
from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from lib.layers import Inception_B, Inception_C, Reduction
from lib.stats_tools import dice_coef, dice_coef_loss, f1_score, f2_score, f05_score

from config import TRAIN_STATS_PICKLE

def make_model(input_shape, learning_rate, learning_rate_decay=0.):
    inputs = Input(input_shape)
    conv1 = Conv2D(32, (3, 3), activation='selu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='selu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
 
    conv2 = Inception_B(64, activation='selu', padding='same')(pool1)
    pool2 = Reduction(16, (2, 2), activation='selu')(conv2)

    conv3 = Inception_B(128, activation='selu', padding='same')(pool2)
    pool3 = Reduction(32, (2, 2))(conv3)

    conv4 = Inception_B(256, activation='selu', padding='same')(pool3)
    pool4 = Reduction(64, (2, 2))(conv4)

    conv5 = Inception_C(512, activation='selu', padding='same')(pool4)

    up6 = Concatenate(axis=3)([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4])
    conv6 = Inception_B(256, activation='selu', padding='same')(up6)

    up7 = Concatenate(axis=3)([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3])
    conv7 = Inception_B(128, activation='selu', padding='same')(up7)

    up8 = Concatenate(axis=3)([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2])
    conv8 = Inception_B(64, activation='selu', padding='same')(up8)
    
    up9 = Concatenate(axis=3)([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1])
    conv9 = Conv2D(32, (3, 3), activation='selu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='selu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    
    model.compile(optimizer=Adam(lr=learning_rate, decay=learning_rate_decay),
                  loss=dice_coef_loss,
                  metrics=[dice_coef, f1_score, f2_score, f05_score])
    
    return model



class NerveSegmentation():

    def __init__(self, weights_file):
        self.model = load_model(weights_file,
                       custom_objects={'dice_coef_loss': dice_coef_loss, 
                                       'dice_coef': dice_coef,
                                       'f1_score': f1_score,
                                       'f2_score': f2_score,
                                       'f05_score': f05_score})

        self.normalize_ = self.make_normalizer_()



    def predict(self, X, batch_size=1):
        return self.model.predict(self.normalize_(X), batch_size=batch_size)

    def make_normalizer_(self):
        if os.path.exists(TRAIN_STATS_PICKLE):
            with open(TRAIN_STATS_PICKLE, 'rb') as f:
                stats = pickle.load(f)
            mean = stats['mean']
            std = stats['std']
            normalize = lambda imgs: (imgs - mean) / std
        else:
            normalize = lambda imgs: (imgs - imgs.mean()) / imgs.std()

        return lambda X: normalize(X)
