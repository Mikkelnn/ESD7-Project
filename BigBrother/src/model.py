from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf #noqa


def residual_block_v2(x, filters, kernel_size=(3,3), reduction=8):
    shortcut = x

    # --- Project shortcut if channel mismatch ---
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1,1), padding='same', use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # --- Conv 1 ---
    y = Conv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    # --- Conv 2 ---
    y = Conv2D(filters, kernel_size, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)

    # --- Squeeze & Excitation ---
    se = GlobalAveragePooling2D()(y)
    se = Dense(filters // reduction, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1,1,filters))(se)
    y = Multiply()([y, se])

    # --- Merge + final activation ---
    out = Add()([y, shortcut])
    out = Activation('relu')(out)

    return out

def residual_block_v1(x, filters, kernel_size=(3,3)):
    shortcut = x

    # If channels do NOT match, project shortcut
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1,1), padding='same', activation=None)(shortcut)

    x = Conv2D(filters, kernel_size, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    # add shortcut
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def softargmax2d(heatmap):
    # heatmap: (B, H, W, 1)

    shape = tf.shape(heatmap)
    B = shape[0]
    H = shape[1]
    W = shape[2]

    # reshape to (B, H*W)
    flat = tf.reshape(heatmap, (B, H*W))
    flat = tf.nn.softmax(flat, axis=-1)

    # coordinate grids
    xs = tf.linspace(0.0, 1.0, W)
    ys = tf.linspace(0.0, 1.0, H)

    # meshgrid -> (H, W)
    xs, ys = tf.meshgrid(xs, ys)

    xs = tf.reshape(xs, (H*W,))
    ys = tf.reshape(ys, (H*W,))

    # expected coordinate
    x = tf.reduce_sum(flat * xs, axis=1, keepdims=True)
    y = tf.reduce_sum(flat * ys, axis=1, keepdims=True)

    return tf.concat([x, y], axis=1)

class ResidualBlockV2Layer(Layer):
    def __init__(self, filters, kernel_size=(3,3), reduction=8):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.reduction = reduction

        # Main path
        self.conv1 = Conv2D(filters, kernel_size, padding='same', use_bias=False)
        self.bn1   = BatchNormalization()
        self.act1  = Activation('relu')

        self.conv2 = Conv2D(filters, kernel_size, padding='same', use_bias=False)
        self.bn2   = BatchNormalization()

        # Squeeze & Excitation
        self.gap   = GlobalAveragePooling2D()
        self.fc1   = Dense(filters // reduction, activation='relu')
        self.fc2   = Dense(filters, activation='sigmoid')

        # Shortcut projection (only created if needed)
        self.shortcut_conv = None
        self.shortcut_bn   = None

    def build(self, input_shape):
        in_channels = int(input_shape[-1])

        # Create projection shortcut if channels mismatch
        if in_channels != self.filters:
            self.shortcut_conv = Conv2D(self.filters, (1,1), padding='same', use_bias=False)
            self.shortcut_bn   = BatchNormalization()

        super().build(input_shape)

    def call(self, inputs, training=None):
        shortcut = inputs

        # --- Shortcut projection ---
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_bn(shortcut, training=training)

        # --- Conv 1 ---
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)

        # --- Conv 2 ---
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        # --- Squeeze & Excitation ---
        se = self.gap(x)
        se = self.fc1(se)
        se = self.fc2(se)
        se = Reshape((1,1,self.filters))(se)
        x = Multiply()([x, se])

        # --- Merge + ReLU ---
        out = Add()([x, shortcut])
        out = Activation('relu')(out)

        return out

    def compute_output_shape(self, input_shape):
        # Residual block preserves size, only channels may change
        return input_shape[:-1] + (self.filters,)

def define_sweep_single_localization():
    inputs = Input(shape=(21, 1024, 256, 1))

    # ---- per-beam CNN ----------
    x = TimeDistributed(Conv2D(32, (3,3), padding='same', activation='relu'))(inputs)
    x = TimeDistributed(Conv2D(32, (3,3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(16, (3,3), padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D((2,1)))(x)   # 1024→512

    x = TimeDistributed(Conv2D(64, (3,3), padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)   # 512→256

    # residual blocks must accept a tensor of shape (None, None, None, C)
    # x = TimeDistributed(ResidualBlockV2Layer(64))(x)
    x = TimeDistributed(Conv2D(64, (3,3), padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)   # 256→128

    # x = TimeDistributed(ResidualBlockV2Layer(64))(x)
    x = TimeDistributed(Conv2D(64, (3,3), padding='same', activation='relu'))(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x)   # 128→64

    # ---- fuse 21 beams ----------
    # Best for faint targets → picks the beam with best SNR
    x = Lambda(lambda t: tf.reduce_max(t, axis=1))(x)  
    # Now shape is (64, 128, C)

    # --- Coordinate head ---
    heatmap = Conv2D(64, (3,3), padding='same', activation='relu')(x)
    heatmap = Conv2D(64, (3,3), padding='same', activation='relu')(heatmap)
    heatmap = Conv2D(32, (3,3), padding='same', activation='relu')(heatmap)
    heatmap = Conv2D(1, (1,1), activation='relu', padding='same')(heatmap)
    coords = Lambda(softargmax2d)(heatmap)

    # small correction head
    coords = Dense(32, activation='relu')(coords)
    coords = Dense(16, activation='relu')(coords)
    coords_out = Dense(2, activation='sigmoid')(coords)

    model = Model(inputs, coords_out)
    return model

def define_sweep_single_localization_lstm_first(
    time_steps=21,
    range_bins=256,
    vel_bins=1024,
    range_min_m=0.0,
    range_max_m=1000.0,
    vel_min_mps=-2694.0,
    vel_max_mps=2694.0,
):
    inp = Input(
        shape=(time_steps, vel_bins, range_bins, 1),
        name="radar_sequence"
    )

    # Spatiotemporal integration
    x = ConvLSTM2D(
        filters=8,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="tanh"
    )(inp)

    x = BatchNormalization()(x)

    x = ConvLSTM2D(
        filters=16,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=False,
        activation="tanh"
    )(x)

    # Shared logits
    logits = Conv2D(
        filters=1,
        kernel_size=(1,1),
        padding="same",
        activation=None
    )(x)

    # Range marginalization
    range_logits = Lambda(
        lambda t: tf.reduce_sum(t, axis=1),
        name="range_logits"
    )(logits)
    range_logits = Flatten()(range_logits)
    range_prob = Softmax(name="range_prob")(range_logits)

    # Velocity marginalization
    vel_logits = Lambda(
        lambda t: tf.reduce_sum(t, axis=2),
        name="vel_logits"
    )(logits)
    vel_logits = Flatten()(vel_logits)
    vel_prob = Softmax(name="vel_prob")(vel_logits)

    # Bin indices (constants)
    r_idx = tf.range(range_bins, dtype=tf.float32)
    v_idx = tf.range(vel_bins, dtype=tf.float32)

    # Expected bin (soft-argmax)
    r_bin = Lambda(
        lambda t: tf.reduce_sum(t * r_idx[None, :], axis=1),
        name="range_bin"
    )(range_prob)

    v_bin = Lambda(
        lambda t: tf.reduce_sum(t * v_idx[None, :], axis=1),
        name="vel_bin"
    )(vel_prob)

    # Sub-bin offsets
    g = GlobalAveragePooling2D()(x)
    g = Dense(64, activation="relu")(g)
    offsets = Dense(2, activation="tanh", name="offsets")(g)

    # Physical conversion
    range_bin_width = (range_max_m - range_min_m) / float(range_bins)
    vel_bin_width = (vel_max_mps - vel_min_mps) / float(vel_bins)

    range_m = Lambda(
        lambda t: (t[0] + t[1][:, 0]) * range_bin_width + range_min_m,
        name="range_m"
    )([r_bin, offsets])

    vel_mps = Lambda(
        lambda t: (t[0] + t[1][:, 1]) * vel_bin_width + vel_min_mps,
        name="vel_mps"
    )([v_bin, offsets])

    model = Model(inp, [range_m, vel_mps])
    return model

def define_sweep_single_localization_lstm(
    time_steps=21,
    range_bins=256,
    vel_bins=1024,
    range_min_m=0.0,
    range_max_m=1000.0,
    vel_min_mps=-2694.0,
    vel_max_mps=2694.0,
):
    inp = Input(
        shape=(time_steps, vel_bins, range_bins, 1),
        name="radar_sequence"
    )

    # --- EARLY DOWNSAMPLING (CRITICAL) ---
    x = TimeDistributed(
        Conv2D(
            filters=8,
            kernel_size=3,
            strides=2,
            padding="same",
            activation="relu"
        )
    )(inp)

    # --- Spatiotemporal integration ---
    x = ConvLSTM2D(
        filters=8,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        activation="tanh"
    )(x)

    # REMOVE BatchNorm (kills temporal signal)

    x = ConvLSTM2D(
        filters=16,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=False,
        activation="tanh"
    )(x)

    # --- Shared logits ---
    logits = Conv2D(
        filters=1,
        kernel_size=(1, 1),
        padding="same",
        activation=None
    )(x)

    # --- PROPER MARGINALIZATION (log-sum-exp) ---
    range_logits = Lambda(
        lambda t: tf.reduce_logsumexp(t, axis=1),
        name="range_logits"
    )(logits)
    range_logits = Flatten()(range_logits)
    range_prob = Softmax(name="range_prob")(range_logits)

    vel_logits = Lambda(
        lambda t: tf.reduce_logsumexp(t, axis=2),
        name="vel_logits"
    )(logits)
    vel_logits = Flatten()(vel_logits)
    vel_prob = Softmax(name="vel_prob")(vel_logits)

    # --- Bin indices ---
    r_idx = tf.range(range_bins, dtype=tf.float32)
    v_idx = tf.range(vel_bins, dtype=tf.float32)

    # --- Soft-argmax ---
    r_bin = Lambda(
        lambda t: tf.reduce_sum(t * r_idx[None, :], axis=1),
        name="range_bin"
    )(range_prob)

    v_bin = Lambda(
        lambda t: tf.reduce_sum(t * v_idx[None, :], axis=1),
        name="vel_bin"
    )(vel_prob)

    # --- Sub-bin offsets (CLAMPED) ---
    g = GlobalAveragePooling2D()(x)
    g = Dense(64, activation="relu")(g)

    offsets = Dense(2, activation="tanh")(g)
    offsets = Lambda(lambda t: 0.5 * t, name="offsets")(offsets)

    # --- Physical conversion ---
    range_bin_width = (range_max_m - range_min_m) / float(range_bins)
    vel_bin_width = (vel_max_mps - vel_min_mps) / float(vel_bins)

    range_m = Lambda(
        lambda t: (t[0] + t[1][:, 0]) * range_bin_width + range_min_m,
        name="range_m"
    )([r_bin, offsets])

    vel_mps = Lambda(
        lambda t: (t[0] + t[1][:, 1]) * vel_bin_width + vel_min_mps,
        name="vel_mps"
    )([v_bin, offsets])

    model = Model(inp, [range_m, vel_mps])
    return model


def define_robust_model_v2(use_heatmap=True):
    inputs = Input(shape=(1024, 256, 1))

    # --- Initial conv layers ---
    x = Conv2D(32, (3,3), padding='same', activation='relu')(inputs)
    x = Conv2D(32, (3,3), padding='same', activation='relu')(x)    
    x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,1))(x)  # (1024,256)->(512,256)

    x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    # --- Residual blocks ---
    x = residual_block_v2(x, 64)
    x = MaxPooling2D((2,2))(x)

    x = residual_block_v2(x, 128)
    x = MaxPooling2D((2,2))(x)

    # --- Flatten for dense layers ---
    flat = Flatten()(x)

    # --- Target presence head ---
    target_out_head = Dense(256, activation='relu')(flat)
    target_out = Dense(1, activation='sigmoid', name="target_present")(target_out_head)

    # --- Coordinate head ---
    if use_heatmap:
        # Predict 2D heatmap
        heatmap = Conv2D(64, (3,3), padding='same', activation='relu')(x)
        heatmap = Conv2D(64, (3,3), padding='same', activation='relu')(heatmap)
        heatmap = Conv2D(32, (3,3), padding='same', activation='relu')(heatmap)
        heatmap = Conv2D(1, (1,1), activation='relu', padding='same')(heatmap)
        coords_out = Lambda(softargmax2d, name="coords")(heatmap)
    else:
        # Direct regression
        shared_dense = Dense(256, activation='relu')(flat)
        shared_dense = Dense(256, activation='relu')(shared_dense)
        coords_out = Dense(2, activation='sigmoid', name="coords")(shared_dense)

    model = Model(inputs, [target_out, coords_out])
    return model

def define_robust_model_v1(use_heatmap=True):
    inputs = Input(shape=(1024, 256, 1))

    # --- Initial conv layers ---
    x = Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2,1))(x)  # (1024,256)->(512,256)

    x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)

    # --- Residual blocks ---
    x = residual_block_v1(x, 64)
    x = MaxPooling2D((2,2))(x)

    x = residual_block_v1(x, 128)
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
        coords_out = Lambda(softargmax2d, name="coords")(heatmap)
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


def defineModel_single_target_detector_sweep_small():

    inputs = Input(shape=(21,1024,256,1))

    # per-frame CNN
    x = TimeDistributed(Conv2D(4,(3,3),padding="same",activation="relu"))(inputs)
    x = TimeDistributed(MaxPooling2D((2,1)))(x) # (1024, 256) -> (512, 256)

    x = TimeDistributed(Conv2D(8,(3,3),padding="same",activation="relu"))(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x) # (512, 256) -> (256, 128)

    # flatten every frame
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    # fuse angle information: simple max or mean → lightweight and effective
    x = GlobalMaxPooling1D()(x)

    # classifier
    x = Dense(128,activation="relu")(x)
    outputs = Dense(2,activation="softmax")(x)

    model = Model(inputs, outputs)
    return model

def defineModel_single_target_detector_sweep():

    inputs = Input(shape=(21,1024,256,1))

    # per-frame CNN
    x = TimeDistributed(Conv2D(16,(3,3),padding="same",activation="relu"))(inputs)
    x = TimeDistributed(MaxPooling2D((2,1)))(x) # (1024, 256) -> (512, 256)

    x = TimeDistributed(Conv2D(32,(3,3),padding="same",activation="relu"))(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x) # (512, 256) -> (256, 128)

    x = TimeDistributed(Conv2D(64,(3,3),activation="relu",padding="same"))(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x) # (256, 128) -> (128, 64)

    x = TimeDistributed(Conv2D(128,(3,3),activation="relu",padding="same"))(x)
    x = TimeDistributed(MaxPooling2D((2,2)))(x) # (128, 64) -> (64, 32)

    # flatten every frame
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    # fuse angle information: simple max or mean → lightweight and effective
    x = GlobalMaxPooling1D()(x)

    # classifier
    x = Dense(256,activation="relu")(x)
    outputs = Dense(2,activation="softmax")(x)

    model = Model(inputs, outputs)
    return model

def defineModel_single_target_detector_frame(activateion="sigmoid"):

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
        Dense(2, activation=activateion), #TODO Potentially use tanh activation of -1 to 1
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

def defineModel_single_target_detector_doubleConv():

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

