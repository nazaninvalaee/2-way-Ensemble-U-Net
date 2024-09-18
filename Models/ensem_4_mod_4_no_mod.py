from tensorflow.keras import models, layers
from Models import layer_4_mod
from Models import layer_4_no_mod

# Channel Attention Block for the Ensemble
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

    cbam_feature = layers.Add()([avg_pool, max_pool])
    cbam_feature = layers.Multiply()([input_feature, cbam_feature])

    return cbam_feature

# Ensemble model
def create_model():

    # Load 4 layer mod and no mod models
    model1 = layer_4_mod.create_model(1)
    model2 = layer_4_no_mod.create_model(1)

    # Input
    inp = layers.Input(shape=(256, 256, 1))

    # Get outputs from both models
    out1 = model1(inp)
    out2 = model2(inp)

    # Concatenate the outputs from both models
    conc1 = layers.concatenate([out2, out1])

    # Apply attention mechanism to the concatenated output
    conc1 = channel_attention(conc1)

    # Further refinement with convolution layers
    conv2 = layers.Conv2D(16, 3, activation='relu', padding='same')(conc1)
    conv2 = layers.Conv2D(16, 3, activation='relu', padding='same')(conv2)

    # Add a final output layer
    outp1 = layers.Conv2D(8, 1, name='output1', activation='sigmoid', padding='same')(conv2)

    # Create the ensemble model
    model = models.Model(inputs=inp, outputs=outp1)

    return model
