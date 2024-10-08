from tensorflow.keras import models, layers
from tensorflow.keras.layers import Add, Multiply

# Residual Block
def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding='same', activation='relu', use_bias=False)(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    # If the number of filters does not match, use Conv2D to match
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same', use_bias=False)(shortcut)
    x = Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

# Multi-Scale Feature Extraction
def multi_scale_conv(x, filters):
    conv1 = layers.Conv2D(filters, 3, padding='same', activation='relu', use_bias=False)(x)
    conv2 = layers.Conv2D(filters, 5, padding='same', activation='relu', use_bias=False)(x)
    conv3 = layers.Conv2D(filters, 7, padding='same', activation='relu', use_bias=False)(x)
    concatenated = layers.Concatenate()([conv1, conv2, conv3])  # 3*filters
    # Reduce back to 'filters' using 1x1 convolution
    reduced = layers.Conv2D(filters, 1, padding='same', activation='relu', use_bias=False)(concatenated)
    return reduced

# Updated Model with Residual Blocks and Multi-Scale Convolutions
def create_model(ensem=0):
    inp = layers.Input(shape=(256, 256, 1))

    # Downsample path
    conv1 = multi_scale_conv(inp, 16)  # 16 filters
    conv1 = residual_block(conv1, 16)
    pool1 = layers.MaxPool2D(2)(conv1)

    conv2 = multi_scale_conv(pool1, 32)
    conv2 = residual_block(conv2, 32)
    pool2 = layers.MaxPool2D(2)(conv2)

    conv3 = multi_scale_conv(pool2, 64)
    conv3 = residual_block(conv3, 64)
    pool3 = layers.MaxPool2D(2)(conv3)

    conv4 = multi_scale_conv(pool3, 128)
    conv4 = residual_block(conv4, 128)
    pool4 = layers.MaxPool2D(2)(conv4)

    bottleneck = multi_scale_conv(pool4, 256)
    bottleneck = residual_block(bottleneck, 256)

    # No attention blocks in the no_mod model

    # Upsample path
    up4 = layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same', use_bias=False)(bottleneck)
    concat4 = layers.concatenate([up4, conv4])
    up4_conv = residual_block(concat4, 128)

    up3 = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same', use_bias=False)(up4_conv)
    concat3 = layers.concatenate([up3, conv3])
    up3_conv = residual_block(concat3, 64)

    up2 = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same', use_bias=False)(up3_conv)
    concat2 = layers.concatenate([up2, conv2])
    up2_conv = residual_block(concat2, 32)

    up1 = layers.Conv2DTranspose(16, 3, strides=2, activation='relu', padding='same', use_bias=False)(up2_conv)
    concat1 = layers.concatenate([up1, conv1])
    up1_conv = residual_block(concat1, 16)

    if ensem == 1:
        model = models.Model(inputs=inp, outputs=up1_conv)
    else:
        final_output = layers.Conv2D(8, 1, activation='sigmoid', padding='same')(up1_conv)
        model = models.Model(inputs=inp, outputs=final_output)

    return model
