import sys
import csv
import math
import numpy as np
import warnings

from scipy.stats import pearsonr
from collections import defaultdict
from pprint import pprint

original_file_path = '/home/kenelly/workspaces/emotionnewsheadlines/emotiondetection/resources/semeval2007datavalid.csv'
predicted_file_path = '/home/kenelly/workspaces/emotionnewsheadlines/fasttext/fastText-0.1.0/model_epoch50lr1dim300bigram_ml_output.txt'
# predicted_file_path = ''
pearson_calc_mode = 'row'  # row or column
row_threshold = 50 # -1 for accepting all lines

# this order will be kept in all arrays
emotions = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']


def load_original():
    original = []
    gold_standard_label = []
    with open(original_file_path) as f:
        for row in csv.DictReader(f):
            emotion_line = []
            for emotion in emotions:
                emotion_line.append(float(row[emotion]))
            if len([e for e in emotion_line if e >= row_threshold]) > 0:
                original.append(emotion_line)
                gold_standard_label.append(emotions[np.argmax(emotion_line)])
    return original, gold_standard_label


def load_predicted():
    predicted = []
    with open(predicted_file_path) as f:
        for line in f:
            emotion_line = []
            emotions_line = [e.replace('__label__', '') for i, e in enumerate(line.split()) if i%2 == 0]
            scores_line = [e for i, e in enumerate(line.split()) if i%2 == 1]
            d = dict(zip(emotions_line, scores_line))
            for emotion in emotions:
                emotion_line.append(float(d[emotion]))

            predicted.append(emotion_line)
    return predicted


def norm(arr):
    def avoid_non_zero(value, row_sum):
        if row_sum == 0:
            return 0
        return value/float(row_sum)

    normalized_arr = []
    for row in arr:
        s = sum(row)
        normalized_row = [avoid_non_zero(e, float(s)) for e in row]
        normalized_arr.append(normalized_row)
    return normalized_arr


def print_summary(arr):
    print('Summary for winner labels in original file:')
    for emotion in emotions:
        print('\t{}: {}'.format(emotion, arr.count(emotion)))
    print('')


def calc_pearson_row(y_true, y_pred, gold_standard_label):
    pearson = [pearsonr(t, p)[0] for t, p in zip(y_true, y_pred)]
    pearson_per_label = defaultdict(float)
    counter_per_label = defaultdict(int)
    for pearson_score, emotion in zip(pearson, gold_standard_label):
        if not math.isnan(pearson_score):
            pearson_per_label[emotion] += pearson_score
            counter_per_label[emotion] += 1

    for k, v in pearson_per_label.items():
        pearson_per_label[k] = v/float(counter_per_label[k])
    return pearson_per_label


def calc_pearson_column(y_true, y_pred):
    pearson_per_label = defaultdict(float)
    y_true_np_array = np.array(y_true)
    y_pred_np_array = np.array(y_pred)
    for i, emotion in enumerate(emotions):
        pearson_per_label[emotion] = pearsonr(y_true_np_array[:, i],
                                              y_pred_np_array[:, i])[0]
    return pearson_per_label


def calc_pearson(y_true, y_pred, gold_standard_label):
    if pearson_calc_mode == 'row':
        return calc_pearson_row(y_true, y_pred, gold_standard_label)
    elif pearson_calc_mode == 'column':
        return calc_pearson_column(y_true, y_pred)
    else:
        print('Unknown {} mode.'.format(pearson_calc_mode))
        sys.exit(1)


def main():
    print('Loading original and predicted files')
    original, gold_standard_label = load_original()
    predicted = load_predicted()

    if len(original) != len(predicted):
        print('Error: original and predicted does no have the same length! '
              'Original = {} Predicted = {}'.format(len(original), len(predicted)))
        sys.exit(1)

    print_summary(gold_standard_label)

    print('Normalizing only original scores')
    original_normalized = norm(original)

    print('Calculating Pearson score per example and averaging per label')
    pearson_per_label = calc_pearson(original_normalized, predicted, gold_standard_label)
    pprint(pearson_per_label)


if __name__== '__main__':
    warnings.simplefilter("ignore")
    main()