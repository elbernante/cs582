import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def image_augment(X, y, batch_size=32, seed=1):        
    # Create two instances with the same arguments
    data_gen_args = dict(rotation_range=30.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.1,
                         horizontal_flip=True,
                         vertical_flip=True,
                         fill_mode='constant',
                         cval=0)
    
    img_gen = ImageDataGenerator(**data_gen_args)
    mask_gen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    images = img_gen.flow(x=X, y=None, seed=seed, batch_size=batch_size)
    masks = mask_gen.flow(x=y, y=None, seed=seed, batch_size=batch_size)

    # combine generators into one which yields image and masks
    def generator():
        for x_o, y_o in zip(images, masks):
            if (x_o.shape[0]) < batch_size:
                continue
            yield x_o, y_o
    
    return generator()


def get_train_val_sets(train_set_npz, val_set_npz):
    if os.path.isfile(train_set_npz) and os.path.isfile(val_set_npz):
        print("\nArchived data set found.")
        print("... loading {}".format(train_set_npz))
        t_set = np.load(train_set_npz)
        t_x = t_set['x']
        t_y = t_set['y']
        print("... loading {}".format(val_set_npz))
        v_set = np.load(val_set_npz)
        v_x = v_set['x']
        v_y = v_set['y']
    else:
        raise FileNotFoundError("Can't find dataset: " + train_set_npz 
                                + " or " + val_set_npz +
                                ". Run preprocess.py first.")
    
    return t_x, t_y, v_x, v_y