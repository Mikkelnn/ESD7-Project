from tensorflow.keras.layers import Layer, Conv2D, Input, MaxPooling2D, Convolution2D, Flatten, Dense, Reshape, InputLayer, GlobalAveragePooling2D, Add, Activation, BatchNormalization, Reshape, Lambda
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf #noqa


def residual_block(x, filters, kernel_size=(3,3)):
    shortcut = x
    x = Conv2D(filters, kernel_size, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    # add shortcut
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def soft_argmax_2d(heatmap):
    """
    Converts a 2D heatmap to normalized coordinates [0,1].
    heatmap: (batch, H, W, 1)
    Returns (batch, 2) -> [row_norm, col_norm]
    """
    heatmap = tf.squeeze(heatmap, axis=-1)  # shape (batch, H, W)
    heatmap = tf.nn.softmax(tf.reshape(heatmap, [tf.shape(heatmap)[0], -1]), axis=-1)
    H = tf.shape(heatmap)[1]
    W = tf.shape(heatmap)[2] if len(heatmap.shape) > 2 else tf.shape(heatmap)[1]
    heatmap_2d = tf.reshape(heatmap, [tf.shape(heatmap)[0], H, W])
    # row and col grids
    rows = tf.linspace(0.0, 1.0, H)
    cols = tf.linspace(0.0, 1.0, W)
    row_coords = tf.reduce_sum(tf.reduce_sum(heatmap_2d, axis=2) * rows[None, :], axis=1)
    col_coords = tf.reduce_sum(tf.reduce_sum(heatmap_2d, axis=1) * cols[None, :], axis=1)
    return tf.stack([row_coords, col_coords], axis=-1)

def define_robust_model(use_heatmap=True):
    inputs = Input(shape=(1024, 256, 1))

    # --- Initial conv layers ---
    x = Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2,1))(x)  # (1024,256)->(512,256)

    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    # --- Residual blocks ---
    x = residual_block(x, 64)
    x = MaxPooling2D((2,2))(x)

    x = residual_block(x, 128)
    x = MaxPooling2D((2,2))(x)

    # --- Flatten for dense layers ---
    flat = Flatten()(x)
    shared_dense = Dense(256, activation='relu')(flat)
    shared_dense = Dense(256, activation='relu')(shared_dense)

    # --- Target presence head ---
    target_out = Dense(1, activation='sigmoid', name="target_present")(shared_dense)

    # --- Coordinate head ---
    if use_heatmap:
        # Predict 2D heatmap
        heatmap = Conv2D(1, (1,1), activation='relu', padding='same')(x)
        coords_out = Lambda(soft_argmax_2d, name="coords")(heatmap)
    else:
        # Direct regression
        coords_out = Dense(2, activation='sigmoid', name="coords")(shared_dense)

    model = Model(inputs, [target_out, coords_out])
    return model


class SoftArgmax2D(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, heatmap):
        # heatmap: (B, H, W, 1)
        heatmap = tf.squeeze(heatmap, axis=-1)               # (B, H, W)
        prob = tf.nn.softmax(tf.reshape(heatmap, [tf.shape(heatmap)[0], -1]), axis=-1)
                                                             # (B, H*W)

        H = tf.shape(heatmap)[1]
        W = tf.shape(heatmap)[2]

        # coordinate grid
        xs = tf.linspace(0.0, 1.0, W)
        ys = tf.linspace(0.0, 1.0, H)
        grid_x, grid_y = tf.meshgrid(xs, ys)

        grid = tf.stack([tf.reshape(grid_y, [-1]), tf.reshape(grid_x, [-1])], axis=1)
        # shape (H*W, 2) → (range, doppler) in normalized [0..1]

        coords = tf.matmul(prob, tf.cast(grid, tf.float32))  # (B, 2)
        return coords
    
def defineModel_singel_target_estimate():
    inputs = Input(shape=(1024, 256, 1))

    x = Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2,1))(x)   # (1024,256) -> (512,256)

    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)   # (512,256) -> (256,128)

    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)   # (256,128) -> (128,64)

    x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)   # (128,64) -> (64,32)

    # --- Heatmap head ---
    heatmap = Conv2D(1, (1,1), activation=None, name="heatmap_logits")(x)
    # shape: (batch, 64, 32, 1)

    # --- Soft-argmax localization (normalized [0..1] range+doppler) ---
    coords = SoftArgmax2D(name="coords")(heatmap)

    # coords = (batch, 2) → [range_norm, doppler_norm]

    # --- Target-present head (use pooled CNN features) ---
    p = GlobalAveragePooling2D()(x)
    target_out = Dense(1, activation='sigmoid', name="target_present")(p)

    # Final model
    model = Model(inputs, [target_out, coords]) # [target_out, coords, heatmap]
    return model

def defineModel_singel_target_estimate_descreete(range_bins, doppler_bins):
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
        
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 1)),  # (1024, 256) -> (512, 256)

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),  # (512, 256) -> (256, 128)

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),  # (256, 128) -> (128, 64)

        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(pool_size=(2, 2)),  # (128, 64) -> (64, 32)
        
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

