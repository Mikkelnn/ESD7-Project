from tensorflow.keras.layers import Conv2D, Input, Dropout, MaxPooling2D, Convolution2D, Flatten, Dense, Reshape, InputLayer, GRU
from tensorflow.keras.models import Sequential, Model


def defineModel_singel_target_estimate(range_bins, doppler_bins):
    inputs = Input(shape=(1024, 256, 1))

    x = Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2,1))(x) # (1024, 256) -> (512, 256)

    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)

    # --- Target-present head ---
    target_out = Dense(1, activation='sigmoid', name="target_present")(x)

    # --- Range head ---
    range_out = Dense(range_bins, activation='softmax', name="range_head")(x)

    # --- Doppler head ---
    doppler_out = Dense(doppler_bins, activation='softmax', name="doppler_head")(x)

    model = Model(inputs, [target_out, range_out, doppler_out])

    return model

def defineModel_smallCNN():

    model = Sequential([
        Input(shape=(1024, 256, 1)),
        Convolution2D(filters=16, kernel_size=(3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 1)),  # (1024, 256) -> (512, 256)
        Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same"),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(2, activation='sigmoid'), #TODO Potentially use tanh activation of -1 to 1
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



def defineModel_single_target_detector():

    model = Sequential([
        Input(shape=(1024, 256, 1)),

        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding="same"),
        Conv2D(32, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 1)),  # (1024, 256) -> (512, 256)

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding="same"),
        Conv2D(64, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),  # (512, 256) -> (256, 128)

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding="same"),
        Conv2D(128, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),  # (256, 128) -> (128, 64)

        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding="same"),
        Conv2D(256, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),  # (128, 64) -> (64, 32)

        Flatten(),
        Dense(256, activation='relu'),
        Dense(256, activation='relu'),
        Dense(2, activation='softmax'),
    ])

    return model

