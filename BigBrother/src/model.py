from tensorflow.keras.layers import Dropout, MaxPooling2D, Convolution2D, Flatten, Dense, InputLayer, GRU
from tensorflow.keras.models import Sequential

def defineModel():

    model = Sequential([
        Convolution2D(filters=8, kernel_size=(3,3), activation='relu', padding="same", input_shape=(1024, 256, 1)),
        MaxPooling2D((2,2), strides=(2,2)), #Half feature set
        Convolution2D(filters=16, kernel_size=(3,3), activation='relu', padding="same"), 
        MaxPooling2D((2,2), strides=(2,2)), #Half feature set
        Convolution2D(filters=32, kernel_size=(3,3), activation='relu', padding="same"),
        MaxPooling2D((2,2), strides=(2,2)), #Half feature set
        Convolution2D(filters=64, kernel_size=(3,3), activation='relu', padding="same"),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(2, activation='sigmoid'), #TODO Potentially use tanh activation of -1 to 1
    ])

    #model = Sequential([
    #    InputLayer(input_shape=(2,)),  # two inputs
    #    Dense(2, use_bias=True),
    #    Dense(2, use_bias=False)
    #])
    # model.add(GRU(units=50)) #Adds Grated Recurrent Units (GRU). This is a subtype of Long/Short- Term Memory (LSTM)
    #Note that this only adds a single layer of n neurons.

    return model