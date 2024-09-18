from tensorflow.keras import models, layers

def create_model(ensem=0):
    inp = layers.Input(shape=(256, 256, 1))

    # Downsample path
    conv1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inp)
    conv1 = layers.Conv2D(16, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPool2D(2)(conv1)

    conv2 = layers.Conv2D(32, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPool2D(2)(conv2)

    conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPool2D(2)(conv3)

    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPool2D(2)(conv4)

    bottleneck = layers.Conv2D(256, 3, activation='relu', padding='same')(pool4)

    # Upsample path
    up4 = layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(bottleneck)
    concat4 = layers.concatenate([up4, conv4])
    up4_conv = layers.Conv2D(128, 3, activation='relu', padding='same')(concat4)

    up3 = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(up4_conv)
    concat3 = layers.concatenate([up3, conv3])
    up3_conv = layers.Conv2D(64, 3, activation='relu', padding='same')(concat3)

    up2 = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(up3_conv)
    concat2 = layers.concatenate([up2, conv2])
    up2_conv = layers.Conv2D(32, 3, activation='relu', padding='same')(concat2)

    up1 = layers.Conv2DTranspose(16, 3, strides=2, activation='relu', padding='same')(up2_conv)
    concat1 = layers.concatenate([up1, conv1])
    up1_conv = layers.Conv2D(16, 3, activation='relu', padding='same')(concat1)

    if ensem == 1:
        model = models.Model(inputs=inp, outputs=up1_conv)
    else:
        final_output = layers.Conv2D(8, 1, activation='sigmoid', padding='same')(up1_conv)
        model = models.Model(inputs=inp, outputs=final_output)

    return model
