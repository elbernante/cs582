from keras.layers import Concatenate, Conv2D, MaxPooling2D, AveragePooling2D

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