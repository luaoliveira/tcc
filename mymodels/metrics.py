# import torch
import numpy as np
import os
import cv2
from pathlib import Path
from graphics import plot_confusion_matrix
from utils import parse_args


def calc_pixel_accuracy(predicted_mask, ground_truth):

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

def calc_confusion_matrix(predicted_mask, ground_truth):

    TP = np.sum((predicted_mask == 255) & (ground_truth == 255))
    FN = np.sum((predicted_mask == 0) & (ground_truth == 255))
    FP = np.sum((predicted_mask == 255) & (ground_truth == 0))
    TN = np.sum((predicted_mask == 0) & (ground_truth == 0))

    return TP, TN, FP, FN

def calc_all_metrics(args):

    output_path = Path("output") / (args["fold_name"] if args.get("fold_name") else args["name"])

    path_pred_masks = output_path / 'result_masks'
    path_ground_truths = Path(args['validation_masks'])
    metrics_file_path = output_path / f'{args["name"]}_test_metrics.txt'

    pred_masks=[]
    ground_truths = []

    files = os.listdir(path_ground_truths)
    n = len(files)

    for file in files:
        print(path_ground_truths, path_pred_masks, file)
        pred_masks.append(cv2.imread(path_pred_masks / file))
        ground_truths.append(cv2.imread(path_ground_truths / file))

    for i in range(n):

        ground_truths[i] = np.where(ground_truths[i] < 30, 0, 255)
        pred_masks[i] = np.where(pred_masks[i] < 30, 0, 255)
        
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
    
    content = [mean_px_acc, mean_iou, mean_precs, mean_recalls, mean_f1s]
    with metrics_file_path.open("w", encoding="utf-8") as file:
        file.write(",".join([name for (name, value) in content]) + "\n")
        file.write(",".join([f"{value:.5f}" for (name, value) in content]) + "\n")
        # for item in content:
            # file.write(f"{item}\n")

    flattened_predm = np.concatenate([img.flatten() for img in pred_masks if img is not None])
    flattened_gts = np.concatenate([img.flatten() for img in ground_truths if img is not None])
    
    # plot_confusion_matrix(flattened_gts, flattened_predm, 'U2NET')
    TP, TN, FP, FN = calc_confusion_matrix(flattened_predm, flattened_gts)
    plot_confusion_matrix(TP, TN, FP, FN, args["name"])

if __name__ == '__main__':
    calc_all_metrics(parse_args())
