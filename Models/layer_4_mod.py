from tensorflow.keras import models, layers
from tensorflow.keras.layers import Add, Multiply

# Channel Attention Block
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_layer_one = layers.Dense(channel // ratio, activation='relu')
    shared_layer_two = layers.Dense(channel, activation='sigmoid')

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Multiply()([input_feature, cbam_feature])

    return cbam_feature

# Spatial Attention Block
def spatial_attention(input_feature):
    avg_pool = layers.Lambda(lambda x: layers.K.mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: layers.K.max(x, axis=-1, keepdims=True))(input_feature)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    spatial_attention = layers.Conv2D(filters=1, kernel_size=7, activation='sigmoid', padding='same')(concat)

    return Multiply()([input_feature, spatial_attention])

# Residual Block
def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

# Multi-Scale Feature Extraction
def multi_scale_conv(x, filters):
    conv1 = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    conv2 = layers.Conv2D(filters, 5, padding='same', activation='relu')(x)
    conv3 = layers.Conv2D(filters, 7, padding='same', activation='relu')(x)
    return layers.Concatenate()([conv1, conv2, conv3])

# Updated Model with Attention, Residual Blocks, and Multi-Scale Convolutions
def create_model(ensem=0):

    inp = layers.Input(shape=(256, 256, 1))

    # Multi-scale feature extraction at the input layer
    conv1 = multi_scale_conv(inp, 16)
    conv1 = residual_block(conv1, 16)
    pool1 = layers.MaxPool2D(2)(conv1)

    conv2 = multi_scale_conv(pool1, 32)
    conv2 = residual_block(conv2, 32)
    pool2 = layers.MaxPool2D(2)(conv2)

    conv3 = multi_scale_conv(pool2, 64)
    conv3 = residual_block(conv3, 64)
    pool3 = layers.MaxPool2D(2)(conv3)

    conv8 = multi_scale_conv(pool3, 128)
    conv8 = residual_block(conv8, 128)
    pool4 = layers.MaxPool2D(2)(conv8)

    # Bottleneck layer
    conv4_ = multi_scale_conv(pool4, 256)
    conv4 = residual_block(conv4_, 256)

    # Apply Channel and Spatial Attention
    conv4 = channel_attention(conv4)
    conv4 = spatial_attention(conv4)

    # Upsampling path
    dconv4 = layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(conv4)
    conc4 = layers.concatenate([dconv4, conv8])
    conv9 = residual_block(conc4, 128)

    dconv3 = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(conv9)
    conc3 = layers.concatenate([dconv3, conv3])
    conv5 = residual_block(conc3, 64)

    dconv2 = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(conv5)
    conc2 = layers.concatenate([dconv2, conv2])
    conv6 = residual_block(conc2, 32)

    dconv1 = layers.Conv2DTranspose(16, 3, strides=2, activation='relu', padding='same')(conv6)
    conc1 = layers.concatenate([dconv1, conv1])
    conv7 = residual_block(conc1, 16)

    if ensem == 1:
        model = models.Model(inputs=inp, outputs=conv7)
    else:
        # Ensure final output has consistent filters (e.g., 16 filters)
        outp1 = layers.Conv2D(16, 1, name='output1', activation='relu', padding='same')(conv7)
        model = models.Model(inputs=inp, outputs=outp1)

    return model
