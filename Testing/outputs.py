import numpy as np
from Testing.evaluation_metrics import *
from skimage.segmentation import find_boundaries
from tqdm import tqdm

# Function for getting the class for each pixel and calculating TP, FP, FN counts for each class
def get_count(output, actual=[]):
  
    req = np.zeros((256, 256), dtype=np.uint8)  # Variable to store predicted classes
    out = output[0].reshape((256, 256, 8))

    tp = np.zeros(8)
    fp = np.zeros(8)
    fn = np.zeros(8)

    for i in range(256):
        for j in range(256):
            a = out[i, j, :].tolist()  # Probability values for each class
            req[i, j] = a.index(max(a))  # Class with the highest probability

            if list(actual):
                if req[i, j] == actual[i, j]:
                    tp[req[i, j]] += 1
                else:
                    fp[req[i, j]] += 1
                    fn[actual[i, j]] += 1

    if not list(actual):
        return req

    return tp, fp, fn, req

# Function for calculating standard evaluation metrics for each class
def calc_met(tp, fp, fn, total, k):
    prec, dice, jac, sens = [], [], [], []

    for i in range(8):
        prec.append(precision(tp[i], fp[i]))
        sens.append(sensitivity(tp[i], fn[i]))
        dice.append(dice_score(tp[i], fp[i], fn[i]))
        jac.append(jaccard(tp[i], fp[i], fn[i]))

    if k == 1:
        return prec, sens, jac, dice

    avg_prec = round(sum(prec) / 8, 2)
    avg_dice = round(sum(dice) / 8, 2)
    avg_jac = round(sum(jac) / 8, 2)
    avg_sens = round(sum(sens) / 8, 2)
    acc = round(accuracy(sum(tp), total), 2)

    if k == 0:
        return avg_prec, avg_sens, avg_jac, avg_dice, acc

# New Function: Calculate Boundary Precision and Recall
def calc_boundary_metrics(pred, actual):
    pred_boundary = find_boundaries(pred, mode='outer')
    actual_boundary = find_boundaries(actual, mode='outer')

    tp = np.sum(np.logical_and(pred_boundary, actual_boundary))
    fp = np.sum(np.logical_and(pred_boundary, np.logical_not(actual_boundary)))
    fn = np.sum(np.logical_and(np.logical_not(pred_boundary), actual_boundary))

    boundary_prec = precision(tp, fp)
    boundary_rec = sensitivity(tp, fn)

    return boundary_prec, boundary_rec

# Function for calculating and printing average and SD of metrics across all classes
def cal_avg_metric(metrics):
    for c, metric in enumerate(metrics, start=1):
        mean_value = round(np.mean(metric), 2)
        std_value = round(np.std(metric), 2)
        print(f"\nMetrics no. {c} for the average of all brain parts")
        print(mean_value, std_value)

# Function for calculating and printing average and SD for each metric for each class
def cal_all_metric(metrics, num):
    for c, metric in enumerate(metrics, start=1):
        print(f"\nMetrics no. {c} for all brain parts\n")
        for i in range(1, 8):
            class_metric = [metric[j][i] for j in range(num)]
            mean_value = round(np.mean(class_metric), 2)
            std_value = round(np.std(class_metric), 2)
            print(mean_value, std_value)

# Main function for prediction, evaluation, and boundary metric calculation
def pred_and_eval(model, X_test, y_test=[], all=0):

    if len(X_test.shape) < 3:  # Handle single test case
        X_test = [X_test]
        if list(y_test):
            y_test = [y_test]

    prec_list, dice_list, jac_list, sens_list, acc_list = [], [], [], [], []
    boundary_prec_list, boundary_rec_list = [], []

    new_out = []
    num = len(X_test)

    for k in tqdm(range(num), desc="Executing", ncols=75):
        output = model.predict(X_test[k].reshape(1, 256, 256))

        if list(y_test):
            actual = y_test[k]

        if not list(y_test):
            new_out.append(get_count(output))

        else:
            tp, fp, fn, out = get_count(output, actual)

            # Calculate metrics if ground truth is provided
            if all == 0:
                prec, sens, jac, dice, acc = calc_met(tp, fp, fn, 256 * 256, all)
                acc_list.append(acc)

            elif all == 1:
                prec, sens, jac, dice = calc_met(tp, fp, fn, 256 * 256, all)

            # Boundary metrics
            boundary_prec, boundary_rec = calc_boundary_metrics(out, actual)
            boundary_prec_list.append(boundary_prec)
            boundary_rec_list.append(boundary_rec)

            prec_list.append(prec)
            dice_list.append(dice)
            jac_list.append(jac)
            sens_list.append(sens)

    # Return new output if no ground truth is provided
    if not list(y_test):
        return new_out

    # Single test case: print results directly
    elif num == 1:
        print('Metric values are: ')
        print(prec_list[0])
        print(sens_list[0])
        print(jac_list[0])
        print(dice_list[0])

        if all == 0:
            print(acc_list[0])

    # Multiple test cases: compute and print average and SD
    elif all == 0:
        cal_avg_metric([prec_list, sens_list, jac_list, dice_list, acc_list])
        cal_avg_metric([boundary_prec_list, boundary_rec_list])

    elif all == 1:
        cal_all_metric([prec_list, sens_list, jac_list, dice_list], num)
        cal_all_metric([boundary_prec_list, boundary_rec_list], num)

