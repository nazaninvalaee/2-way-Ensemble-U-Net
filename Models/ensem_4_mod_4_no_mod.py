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
    out1 = model1(inp)  # Get output from layer_4_mod
    out2 = model2(inp)  # Get output from layer_4_no_mod

    # Print the shapes of out1 and out2 for debugging
    print(f"Shape of out1 (from layer_4_mod): {out1.shape}")
    print(f"Shape of out2 (from layer_4_no_mod): {out2.shape}")

    # Align the number of channels (project to 16 channels)
    out1 = layers.Conv2D(16, 1, activation='relu', padding='same')(out1)
    out2 = layers.Conv2D(16, 1, activation='relu', padding='same')(out2)

    # Print the shapes after projection for verification
    print(f"Shape of out1 after Conv2D(16): {out1.shape}")
    print(f"Shape of out2 after Conv2D(16): {out2.shape}")

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
