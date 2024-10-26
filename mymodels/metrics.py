# import torch
import numpy as np
import os
import cv2

def calc_pixel_accuracy(predicted_mask, ground_truth):

    ground_truth = np.where(ground_truth < 30, 0, 255)
    predicted_mask = np.where(predicted_mask < 30, 0, 255)

    correct_pixels = (predicted_mask == ground_truth).astype(int)
    pixel_accuracy = correct_pixels.sum(axis=None)/correct_pixels.size

    return pixel_accuracy


def calc_iou(predicted_mask, ground_truth):

    predicted_mask = predicted_mask.flatten()
    ground_truth = ground_truth.flatten()

    intersection = np.sum((ground_truth == 255) & (predicted_mask==255))
    union = np.sum((ground_truth == 255) | (predicted_mask==255))

    return intersection/union

def calc_precision(predicted_mask, ground_truth):

    TP = np.sum((predicted_mask == 255) & (ground_truth == 255))

    FP = np.sum((predicted_mask == 255) & (ground_truth == 0))

    return TP/(TP+FP)

def calc_recall(predicted_mask, ground_truth):

    TP = np.sum((predicted_mask == 255) & (ground_truth == 255))

    FN = np.sum((predicted_mask == 0) & (ground_truth == 255))

    return TP/(TP + FN)

def calc_f1_score(predicted_mask, ground_truth):

    precision = calc_precision(predicted_mask, ground_truth)
    recall = calc_recall(predicted_mask, ground_truth)

    return (2*precision*recall)/(precision + recall)


def calc_all_metrics():

    path_pred_masks = 'result_masks/'
    path_ground_truths = 'validation_masks/'

    pred_masks=[]
    ground_truths = []

    files = os.listdir(path_ground_truths)
    n = len(files)

    for file in files:

        pred_masks.append(cv2.imread(f'{path_pred_masks}{file}'))
        ground_truths.append(cv2.imread(f"{path_ground_truths}{file}"))


    for i in range(n):
        
        pixel_accuracies = list(map(calc_pixel_accuracy, pred_masks, ground_truths))
        ious= list(map(calc_iou, pred_masks,ground_truths))
        precisions = list(map(calc_precision, pred_masks, ground_truths))
        recalls = list(map(calc_recall, pred_masks, ground_truths))
        f1_scores = list(map(calc_f1_score, pred_masks, ground_truths))

        ####################

    mean_px_acc = ("Pixel_accuracy", sum(pixel_accuracies)/n)
    mean_iou = ("Intersection over Union", sum(ious)/n)
    mean_precs = ("Precision", sum(precisions)/n)
    mean_recalls = ("Recall", sum(recalls)/n)
    mean_f1s = ("F1 score", sum(f1_scores)/n )

    print(mean_iou)
    print(mean_px_acc)
    print(mean_precs)
    print(mean_recalls)
    print(mean_f1s)
        


if __name__ == '__main__':
    calc_all_metrics()
