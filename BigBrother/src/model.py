from tensorflow.keras.layers import Dropout, MaxPooling2D, Convolution2D, Flatten, Dense, InputLayer, GRU
from tensorflow.keras.models import Sequential



def defineModel():
    # model = Sequential([
    #             Convolution2D(512, kernel_size=(3,3), activation='relu', padding="same", input_shape=(28, 28, 1)),
    #             MaxPooling2D((2,2), strides=(2,2)),
    #             Flatten(),
    #             Dense(10, activation='relu'),
    #             Dropout(0.5)
    #         ])
    model = Sequential([
        InputLayer(input_shape=(2,)),  # two inputs
        Dense(2, use_bias=True),
        Dense(2, use_bias=False)
    ])
    # model.add(GRU(units=50)) #Adds Grated Recurrent Units (GRU). This is a subtype of Long/Short- Term Memory (LSTM)
    #Note that this only adds a single layer of n neurons.

    return model