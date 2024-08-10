from tensorflow.keras import models, layers
from attention_blocks import ChannelAttention, SpatialAttention

def create_model():

    inp = layers.Input(shape=(256, 256, 1))

    conv1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inp)
    conv1 = layers.Conv2D(16, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPool2D(2)(conv1)

    conv2 = layers.Conv2D(32, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv2)
    ca2 = ChannelAttention(conv2)
    conv2 = SpatialAttention(ca2)
    pool2 = layers.MaxPool2D(2)(conv2)

    conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv3)
    ca3 = ChannelAttention(conv3)
    conv3 = SpatialAttention(ca3)
    pool3 = layers.MaxPool2D(2)(conv3)

    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    ca8 = ChannelAttention(conv8)
    conv8 = SpatialAttention(ca8)
    pool4 = layers.MaxPool2D(2)(conv8)

    conv10 = layers.Conv2D(200, 3, activation='relu', padding='same')(pool4)
    conv10 = layers.Conv2D(200, 3, activation='relu', padding='same')(conv10)
    ca10 = ChannelAttention(conv10)
    conv10 = SpatialAttention(ca10)
    pool5 = layers.MaxPool2D(2)(conv10)

    conv4_ = layers.Conv2D(300, 3, activation='relu', padding='same')(pool5)
    conv4 = layers.Conv2D(300, 3, activation='relu', padding='same')(conv4_)

    dconv5 = layers.Conv2DTranspose(200, 3, strides=2, activation='relu', padding='same')(conv4)
    conc5 = layers.concatenate([dconv5, conv10])
    conv11 = layers.Conv2D(200, 3, activation='relu', padding='same')(conc5)
    conv11 = layers.Conv2D(200, 3, activation='relu', padding='same')(conv11)

    dconv4 = layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(conv11)
    conc4 = layers.concatenate([dconv4, conv8])
    conv9 = layers.Conv2D(128, 3, activation='relu', padding='same')(conc4)
    conv9 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv9)

    dconv3 = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(conv9)
    conc3 = layers.concatenate([dconv3, conv3])
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conc3)
    conv5 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv5)

    dconv2 = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(conv5)
    conc2 = layers.concatenate([dconv2, conv2])
    conv6 = layers.Conv2D(32, 3, activation='relu', padding='same')(conc2)
    conv6 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv6)

    dconv1 = layers.Conv2DTranspose(16, 3, strides=2, activation='relu', padding='same')(conv6)
    conc1 = layers.concatenate([dconv1, conv1])
    conv7 = layers.Conv2D(16, 3, activation='relu', padding='same')(conc1)
    conv7 = layers.Conv2D(16, 3, activation='relu', padding='same')(conv7)

    outp1 = layers.Conv2D(8, 1, name='output1', activation='sigmoid', padding='same')(conv7)

    model = models.Model(inputs=inp, outputs=outp1)
    
    return model
