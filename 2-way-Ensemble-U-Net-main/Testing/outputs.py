import numpy as np
from Testing.evaluation_metrics import *
from tqdm import tqdm

def get_count(output, actual=[]):
    """Calculate true positive, false positive, false negative, and predicted class for each pixel."""
    req = np.zeros((256, 256), dtype=np.uint8)
    out = output[0].reshape((256, 256, 8))

    tp = np.zeros(8)
    fp = np.zeros(8)
    fn = np.zeros(8)

    for i in range(256):
        for j in range(256):
            probs = out[i, j, :].tolist()
            req[i, j] = probs.index(max(probs))
            
            if actual:
                if req[i, j] == actual[i, j]:
                    tp[req[i, j]] += 1
                else:
                    fp[req[i, j]] += 1
                    fn[actual[i, j]] += 1

    if not actual:
        return req
    return tp, fp, fn, req

def calc_met(tp, fp, fn, total, k):
    """Calculate evaluation metrics for each class and return average or per-class metrics."""
    prec = [precision(tp[i], fp[i]) for i in range(8)]
    sens = [sensitivity(tp[i], fn[i]) for i in range(8)]
    dice = [dice_score(tp[i], fp[i], fn[i]) for i in range(8)]
    jac = [jaccard(tp[i], fp[i], fn[i]) for i in range(8)]

    if k == 1:
        return prec, sens, jac, dice

    avg_prec = round(sum(prec) / 8, 2)
    avg_sens = round(sum(sens) / 8, 2)
    avg_dice = round(sum(dice) / 8, 2)
    avg_jac = round(sum(jac) / 8, 2)
    acc = round(accuracy(sum(tp), total), 2)

    if k == 0:
        return avg_prec, avg_sens, avg_jac, avg_dice, acc

def cal_avg_metric(metrics):
    """Print the mean and standard deviation of average evaluation metrics over all images."""
    for c, metric in enumerate(metrics, start=1):
        mean_val = round(np.mean(metric), 2)
        std_val = round(np.std(metric), 2)
        print(f'\nMetrics no. {c} for the average of all brain parts')
        print(mean_val, std_val)

def cal_all_metric(metrics, num):
    """Print the mean and standard deviation of all evaluation metrics for each class over all images."""
    for c, metric in enumerate(metrics, start=1):
        print(f'\nMetrics no. {c} for all brain parts\n')
        for i in range(8):
            values = [metric[j][i] for j in range(num)]
            mean_val = round(np.mean(values), 2)
            std_val = round(np.std(values), 2)
            print(mean_val, std_val)

def pred_and_eval(model, X_test, y_test=[], all=0):
    """Predict and evaluate the segmentation model on test images."""
    if X_test.ndim < 3:
        X_test = [X_test]
        y_test = [y_test] if y_test else []

    prec_list, dice_list, jac_list, sens_list, acc_list, new_out = [], [], [], [], [], []
    num = len(X_test)

    for k in tqdm(range(num), desc="Executing", ncols=75):
        output = model.predict(X_test[k].reshape(1, 256, 256))

        if y_test:
            actual = y_test[k]
            tp, fp, fn, _ = get_count(output, actual)

            if all == 0:
                prec, sens, jac, dice, acc = calc_met(tp, fp, fn, 256*256, all)
                acc_list.append(acc)
            elif all == 1:
                prec, sens, jac, dice = calc_met(tp, fp, fn, 256*256, all)

            prec_list.append(prec)
            dice_list.append(dice)
            jac_list.append(jac)
            sens_list.append(sens)
        else:
            new_out.append(get_count(output))

    if not y_test:
        return new_out

    if num == 1:
        print('Metric values are:')
        print(prec_list[0], sens_list[0], jac_list[0], dice_list[0])
        if all == 0:
            print(acc_list[0])
    elif all == 0:
        cal_avg_metric([prec_list, sens_list, jac_list, dice_list, acc_list])
    elif all == 1:
        cal_all_metric([prec_list, sens_list, jac_list, dice_list], num)
