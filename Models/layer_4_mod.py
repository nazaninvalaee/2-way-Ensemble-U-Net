from tensorflow.keras import models, layers
from tensorflow.keras import backend as K


# Channel Attention Block
def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]  # Number of feature maps
    shared_layer_one = layers.Dense(channel // ratio, activation='relu', use_bias=True, kernel_initializer='he_normal')
    shared_layer_two = layers.Dense(channel, use_bias=True, kernel_initializer='he_normal')

    avg_pool = layers.GlobalAveragePooling2D()(input_feature)
    avg_pool = layers.Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = layers.GlobalMaxPooling2D()(input_feature)
    max_pool = layers.Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Activation('sigmoid')(cbam_feature)

    return layers.Multiply()([input_feature, cbam_feature])


# Spatial Attention Block
def spatial_attention(input_feature):
    avg_pool = layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = layers.Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(input_feature)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    cbam_feature = layers.Conv2D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal')(concat)

    return layers.Multiply()([input_feature, cbam_feature])


# Create the model with attention mechanisms
def create_model(ensem=0):
    inp = layers.Input(shape=(256, 256, 1))

    # Encoding path (with attention)
    conv1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inp)
    conv1 = layers.Conv2D(16, 3, activation='relu', padding='same')(conv1)
    conv1 = channel_attention(conv1)  # Apply channel attention here
    conv1 = spatial_attention(conv1)  # Apply spatial attention here
    pool1 = layers.MaxPool2D(2)(conv1)

    conv2 = layers.Conv2D(32, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(32, 3, activation='relu', padding='same')(conv2)
    conv2 = channel_attention(conv2)
    conv2 = spatial_attention(conv2)
    pool2 = layers.MaxPool2D(2)(conv2)

    conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv3)
    conv3 = channel_attention(conv3)
    conv3 = spatial_attention(conv3)
    pool3 = layers.MaxPool2D(2)(conv3)

    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)
    conv8 = channel_attention(conv8)
    conv8 = spatial_attention(conv8)
    pool4 = layers.MaxPool2D(2)(conv8)

    conv4_ = layers.Conv2D(256, 3, activation='relu', padding='same')(pool4)
    conv4 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv4_)
    conv4 = channel_attention(conv4)
    conv4 = spatial_attention(conv4)

    # Decoding path
    dconv4 = layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(conv4)
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

    if ensem == 1:
        model = models.Model(inputs=inp, outputs=conv7)
    else:
        outp1 = layers.Conv2D(8, 1, name='output1', activation='sigmoid', padding='same')(conv7)
        model = models.Model(inputs=inp, outputs=outp1)

    return model
