from tensorflow.keras import models, layers
from Models import layer_4_mod
from Models import layer_4_no_mod
from attention_blocks import channel_attention, spatial_attention

def create_model():
    # 4-layer modified model
    model1 = layer_4_mod.create_model(1)

    # 4-layer non-modified model
    model2 = layer_4_no_mod.create_model(1)

    # Input layer
    inp = layers.Input(shape=(256, 256, 1))

    # Sub-model outputs
    out1 = model1(inp)
    out2 = model2(inp)

    # Concatenate the outputs of both models
    conc1 = layers.concatenate([out2, out1])

    # Apply channel attention
    ca_out = channel_attention(conc1)

    # Apply spatial attention
    sa_out = spatial_attention(ca_out)

    # Additional convolutional layer
    conv2 = layers.Conv2D(16, 3, activation='relu', padding='same')(sa_out)
    outp1 = layers.Conv2D(8, 1, name='output1', activation='sigmoid', padding='same')(conv2)

    # Final model
    model = models.Model(inputs=inp, outputs=outp1)

    return model
