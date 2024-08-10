### Testing

This folder contains files for outputting, testing and evaluating the models.

For testing or getting output from the model, first the model should be loaded, then pred_and_eval() in this folder should be called with the following parameters:
- The model to be used should be passed as the first input
- The input(s) to the model should be passed next
  
For getting output from the model, these 2 inputs are enough. In this case, the model will return a 3D array containing the segmented output to all the 2D input images
  
For evaluating the model's performance, the next parameters should be passed along with the above ones:
- The actual output(s), for the function to compare with
- The last parameter will decide whether to return the average value of the evaluation metrics for all the classes or to return detailed evaluation with respect to each class for all the images in the test set

Evaluation metrics used: Precision, Sensitivity, Jaccard Similarity, Dice Score, Accuracy

Accuracy won't be returned when the detailed evaluation for all the classes are returned.
