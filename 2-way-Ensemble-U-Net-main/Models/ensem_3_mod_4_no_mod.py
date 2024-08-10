from tensorflow.keras import models, layers
from Models import layer_3_mod
from Models import layer_4_no_mod

def create_model():
    # 3-layer modified model
    model1 = layer_3_mod.create_model(1)

    # 4-layer non-modified model
    model2 = layer_4_no_mod.create_model(1)

    # Input layer
    inp = layers.Input(shape=(256, 256, 1))

    # Sub-model outputs
    out1 = model1(inp)
    out2 = model2(inp)

    # Additional layers for ensemble
    out1 = layers.Conv2D(16, 3, activation='sigmoid', padding='same')(out1)
    conc1 = layers.concatenate([out2, out1])
    conv2 = layers.Conv2D(16, 3, activation='relu', padding='same')(conc1)
    outp1 = layers.Conv2D(8, 1, name='output1', activation='sigmoid', padding='same')(conv2)

    # Final model
    model = models.Model(inputs=inp, outputs=outp1)

    return model