from tensorflow.keras.layers import Convolution2D, Flatten, Dense
from tensorflow.keras.layers import Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential



def defineModel():
    model = Sequential([
                Convolution2D(512, kernel_size=(3,3), activation='relu', padding="same", input_shape=(28, 28, 1)),
                MaxPooling2D((2,2), strides=(2,2)),
                Flatten(),
                Dense(10, activation='relu'),
                Dropout(0.5)
            ])

    return model