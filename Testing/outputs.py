import numpy as np
from Testing.evaluation_metrics import *
from tqdm import tqdm
import time





'''
Function for getting the class in which, each pixel of the output belongs, and is stored in 'req' varible.

(The model returns a list of 8 values for each pixel (the probability of pixel belonging to each class,
as the segmentation task was treated as a classication task for each pixel and there are 8 classes in total)

The function then calculates the true positive, false positive and false negative count for each class if the outputs are passed to actual
(by comparing the new output with the ground truth), which are returned along with the new output

'actual' has a default value of -1 to indicate that the 'req' variable is to be returned
'''

def get_count(output, actual=[]):
  
  req = np.zeros((256, 256), dtype=np.uint8)    # Variable in which the new output will be stored
  out = output[0].reshape((256, 256, 8))

  tp = np.zeros(8)
  fp = np.zeros(8)
  fn = np.zeros(8)

  for i in range(256):
    for j in range(256):

      a = out[i, j, :].tolist()     # For each pixel, the 8 probability values are stored in the list a
      req[i, j] = a.index(max(a))   # Returning the index of the list whose value (probability of belonging to that class) is maximum
      
      if list(actual):
        if req[i, j]==actual[i, j]:
          tp[req[i, j]] += 1

        elif (not req[i, j]==actual[i, j]):

          fp[req[i, j]] += 1
          fn[actual[i, j]] += 1
  
  if not list(actual):
    return req

  return tp, fp, fn, req






# Function for calculating the evaluation metrics for each class 
# (using the true positive, false positive and false negative count for each class) 
# Then either those metrics are returned directly, or the average of those metrics, along with the accuracy, are calculated and returned,
# depending on the value of k

def calc_met(tp, fp, fn, total, k):

  prec = []
  dice = []
  jac = []
  sens = []

  for i in range(8):
    prec.append(precision(tp[i], fp[i]))
    sens.append(sensitivity(tp[i], fn[i]))
    dice.append(dice_score(tp[i], fp[i], fn[i]))
    jac.append(jaccard(tp[i], fp[i], fn[i]))

  if k==1:
    return prec, sens, jac, dice

  avg_prec = round(sum(prec)/8, 2)
  avg_dice = round(sum(dice)/8, 2)
  avg_jac = round(sum(jac)/8, 2)
  avg_sens = round(sum(sens)/8, 2)
  acc = round(accuracy(sum(tp), total), 2)

  if k==0:
    return avg_prec, avg_sens, avg_jac, avg_dice, acc






# Function for printing the mean and the standard deviation of the average of the evaluation metric 
# results for all the classes over all the images

def cal_avg_metric(metrics):

  c = 1
  for i in metrics:

    i_mean = round(np.mean(i), 2)
    i_std = round(np.std(i), 2)

    print('\nMetrics no.', c, 'for the average of all brain parts')
    print(i_mean, i_std)
    c += 1





# Function for printing the mean and the standard deviation of all the evaluation metric results for all the classes over all the images

def cal_all_metric(metrics, num):
  # metrics = [prec_list, sens_list, jac_list, dice_list]

  c = 1
  for k in metrics:
    print('\nMetrics no.', c, 'for all brain parts\n')
    c += 1

    for i in range(1, 8):
      l = []

      for j in range(num):
        l.append(k[j][i])

      l_mean = round(np.mean(l), 2)
      l_std = round(np.std(l), 2)
      print(l_mean, l_std)






'''
Function for prediction and evaluation for the segmentation task

For each test image, the segmentation is done then passed on to the two functions:
'get_count' and 'calc_met' for finally getting the evaluation metric values or average values (determined by the 'all' value) per image.

This process is repeated for all the test images to be evaluated.

The values returned by calc_are added to a list for getting the mean and SD of those values over all the test images by:
'cal_avg_metric', for all=0 or 'cal_all_metric', for all=1

This function can also return the segmentation, instead of the metric values, in which case, 
nothing should be passed to the variables 'y_test' and 'all', so they are initialised to their default values
'''

def pred_and_eval(model, X_test, y_test=[], all=0):
  # print(list(y_test))

  if len(X_test.shape)<3:       # For a single test case, converting the 2D slice to a 3D array
    X_test = [X_test]

    if list(y_test):
      y_test = [y_test]

  prec_list = []
  dice_list = []
  jac_list = []
  sens_list = []
  acc_list = []

  new_out = []
  num = len(X_test)   # number of test cases

  for k in tqdm(range(num), desc="Executing", ncols=75):
    output = model.predict(X_test[k].reshape(1, 256, 256))

    if list(y_test):
      actual = y_test[k]

    if not list(y_test):
      new_out.append(get_count(output))

    else:
      tp, fp, fn, out = get_count(output, actual)

      # Calculating the metrics if y_test is not empty, indicating that the metrics are to be calculated

      if all==0:
        prec, sens, jac, dice, acc = calc_met(tp, fp, fn, 256*256, all)
        acc_list.append(acc)

      elif all==1:
        prec, sens, jac, dice = calc_met(tp, fp, fn, 256*256, all)

      prec_list.append(prec)
      dice_list.append(dice)
      jac_list.append(jac)
      sens_list.append(sens)

  print()
  if not list(y_test):
    return new_out

  elif num==1:                          # For num = 1, the metric values will be displayed directly, 
                                        # instead of the mean and SD, since they store only a single value
    print('Metric values are: ')
    
    print(prec_list[0])
    print(sens_list[0])
    print(jac_list[0])
    print(dice_list[0])
   
    if all==0:
      print(acc_list[0])
    
  elif all==0:
    cal_avg_metric([prec_list, sens_list, jac_list, dice_list, acc_list])

  elif all==1:
    cal_all_metric([prec_list, sens_list, jac_list, dice_list], num)



