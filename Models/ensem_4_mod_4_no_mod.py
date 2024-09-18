from tensorflow.keras import models, layers
from Models import layer_4_mod
from Models import layer_4_no_mod

# Ensemble model
def create_model():

    # Load 4 layer mod and no mod models
    model1 = layer_4_mod.create_model(1)
    model2 = layer_4_no_mod.create_model(1)

    # Input
    inp = layers.Input(shape=(256, 256, 1))

    # Get outputs from both models
    out1 = model1(inp)  # Shape (256, 256, 48)
    out2 = model2(inp)  # Shape (256, 256, 16)

    # Align the number of channels (for example, project both to 16 channels)
    out1 = layers.Conv2D(16, 1, activation='relu', padding='same')(out1)  # Shape (256, 256, 16)
    out2 = layers.Conv2D(16, 1, activation='relu', padding='same')(out2)  # Shape (256, 256, 16)

    # Concatenate the outputs from both models
    conc1 = layers.concatenate([out1, out2])

    # Further refinement with convolution layers
    conv2 = layers.Conv2D(16, 3, activation='relu', padding='same')(conc1)
    conv2 = layers.Conv2D(16, 3, activation='relu', padding='same')(conv2)

    # Add a final output layer
    outp1 = layers.Conv2D(8, 1, name='output1', activation='sigmoid', padding='same')(conv2)

    # Create the ensemble model
    model = models.Model(inputs=inp, outputs=outp1)

    return model
