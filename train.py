

from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from keras.callbacks import ModelCheckpoint

from lib.util import image_augment, get_train_val_sets
from lib.callbacks import TrainMonitor

from model import make_model
from config import *

def train():
    
    # Get training and validation data set
    X_train, y_train, X_val, y_val = get_train_val_sets(TRAIN_SET_PICKLE,
                                                        VALIDATION_SET_PICKLE)

    # Create model
    model = make_model((IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
                        LEARNING_RATE,
                        LEARNING_RATE_DECAY)

    # Callbacks for saving the model, and monitoring the tra
    model_checkpoint = ModelCheckpoint(WEIGHTS_FILE, 
                                       monitor='val_loss', save_best_only=True)
    train_monitor = TrainMonitor(HISTORY_LOG, X_train[:1,...], y_train[:1,...], 
                                 out_dir="output/preds/")
    
    # Train model
    return model.fit_generator(image_augment(X_train, y_train, 
                                             batch_size=TRAIN_BATCH_SIZE,
                                             seed=10), 
                               y_train.shape[0] // TRAIN_BATCH_SIZE,
                               epochs=EPOCHS_TO_RUN,
                               verbose=1,
                               callbacks=[model_checkpoint, train_monitor], 
                               validation_data=(X_val, y_val))

if __name__ == "__main__":
    train()


