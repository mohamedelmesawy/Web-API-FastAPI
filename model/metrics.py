import numpy as np
import cv2
from sklearn.metrics import confusion_matrix


def sklearn_compute_IOU(y_pred, y_true, num_classes=7):
     y_pred = y_pred.flatten()
     y_true = y_true.flatten()
     current = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
     # compute mean iou
     intersection = np.diag(current)
     ground_truth_set = current.sum(axis=1)
     predicted_set = current.sum(axis=0)
     union = ground_truth_set + predicted_set - intersection
     IoU = intersection / union.astype(np.float32)
     return np.mean(IoU), IoU


def fast_hist(a, b, n):
        k = (a >= 0) & (a < n)
        return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def compute_IOU(y_pred, y_true, num_classes=7):
    hist = fast_hist(y_true.flatten(), y_pred.flatten(), num_classes)
    IoU_list = per_class_iu(hist)
    present_classes_count = len(IoU_list[IoU_list > 0.009])
    print("--- present_classes_count ------> ", present_classes_count)
    mean_IoU = np.nansum(IoU_list) / present_classes_count
    # mean_IoU = np.nanmean(IoU_list)
    return mean_IoU, IoU_list