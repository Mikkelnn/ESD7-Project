from tensorflow.keras.layers import Dropout, MaxPooling2D, Convolution2D, Flatten, Dense, Reshape, InputLayer, GRU
from tensorflow.keras.models import Sequential

def defineModel(output_size):

    model = Sequential([
        Convolution2D(filters=8, kernel_size=(3,3), activation='sigmoid', padding="same", input_shape=(1024, 256, 1)),
        MaxPooling2D((2,2), strides=(2,2)), #Half feature set
        Convolution2D(filters=16, kernel_size=(3,3), activation='sigmoid', padding="same"), 
        MaxPooling2D((2,2), strides=(2,2)), #Half feature set
        Convolution2D(filters=32, kernel_size=(3,3), activation='sigmoid', padding="same"),
        MaxPooling2D((2,2), strides=(2,2)), #Half feature set
        Convolution2D(filters=64, kernel_size=(3,3), activation='sigmoid', padding="same"),
        Flatten(),
        Dense(256, activation='sigmoid'),
        Dense(256, activation='sigmoid'),
        Dense(output_size, activation='sigmoid'), #TODO Potentially use tanh activation of -1 to 1
    ])

    return model