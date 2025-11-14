from tensorflow.keras.layers import Input, Dropout, MaxPooling2D, Convolution2D, Flatten, Dense, Reshape, InputLayer, GRU
from tensorflow.keras.models import Sequential


def defineModel(range_bins, doppler_bins):
    inputs = Input(shape=(1024, 256, 1))

    x = Conv2D(16, (5,5), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)

    # --- Range head ---
    range_out = Dense(range_bins, activation='softmax', name="range_head")(x)

    # --- Doppler head ---
    doppler_out = Dense(doppler_bins, activation='softmax', name="doppler_head")(x)

    model = Model(inputs, [range_out, doppler_out])

    return model

def defineModel_smallCNN(output_size):

    model = Sequential([
        Input(shape=(1024, 256, 1)),
        Convolution2D(filters=2, kernel_size=(2, 2), activation='relu', padding="same"),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(output_size, activation='sigmoid'), #TODO Potentially use tanh activation of -1 to 1
    ])

    return model


def defineModel_bigCNN(output_size):

    model = Sequential([
        Input(shape=(1024, 256, 1)),
        Convolution2D(filters=1, kernel_size=(80, 20), activation='relu', padding="same"),
        MaxPooling2D((2,2), strides=(2,2)), #Half feature set
        Convolution2D(filters=16, kernel_size=(40,10), activation='relu', padding="same"), 
        MaxPooling2D((2,2), strides=(2,2)), #Half feature set
        Convolution2D(filters=32, kernel_size=(20,5), activation='relu', padding="same"),
        MaxPooling2D((2,2), strides=(2,2)), #Half feature set
        Convolution2D(filters=64, kernel_size=(5,2), activation='relu', padding="same"),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(output_size, activation='sigmoid'), #TODO Potentially use tanh activation of -1 to 1
    ])

    return model