from tensorflow.keras import models, layers
from Models import layer_4_mod
from Models import layer_4_no_mod

def create_model():

    # 4 layer mod
    model1 = layer_4_mod.create_model(1)

    # 4 layer no mod
    model2 = layer_4_no_mod.create_model(1)

    # 4 mod 4 no mod ensemble
    inp = layers.Input(shape=(256, 256, 1))

    # Forward pass through both models
    out1 = model1(inp)  # Output from layer_4_mod with attention mechanisms
    out2 = model2(inp)  # Output from layer_4_no_mod with attention mechanisms

    # Concatenate the outputs from both models
    conc1 = layers.concatenate([out2, out1])

    # Apply a convolutional layer to refine the combined output
    conv2 = layers.Conv2D(16, 3, activation='relu', padding='same')(conc1)
    
    # Final output layer
    outp1 = layers.Conv2D(8, 1, name='output1', activation='sigmoid', padding='same')(conv2)

    # Create the ensemble model
    model = models.Model(inputs=inp, outputs=outp1)

    return model
