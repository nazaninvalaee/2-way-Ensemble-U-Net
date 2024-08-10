### Models

This folder contains files for the models we have developed and tested in our work.

One file is dedicated to each model. Each file contains one function, named 'create_model()', which when called (no parameters need to be passed), will return the respective model.

However, the model will be initialised with random weights, so to use the model for the prediction purpose, the correct weights should be loaded into the model.

This process is a bit lenghty (actually just 3-4 lines of code). A quick alternative can be to directly load the model from the model files, in which case, this module is not needed.

We have included the model weights and the modell files only for the 2-way Ensemble U-Net model. For other models, you need to train them yourselves from scratch, after importing the model using the function.

File named 'ensem_4_mod_4_no_mod.py' is the code for 2-way Ensemble U-Net model.

Follow the demo colab file for the tutorial on how to do it.
