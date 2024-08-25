import numpy as np
from sklearn.metrics import confusion_matrix

import pandas as pd

def get_accuracy(y_test, y_pred):
    """
    calculates the prediction accuracy

    y_test (list or array): true labels
    y_pred (list or array): predicted labels.

    returns accuracy score (float)
    """
    # count the number of correct predictions (TP + TN)
    correct_predictions = np.sum(np.array(y_test) == np.array(y_pred))
    
    # divide by total datapoints (TP + TN + FN + FP)
    accuracy = correct_predictions / len(y_test)
    
    return accuracy


def get_balanced_accuracy(y_test, y_pred):
    """
    calculates the balanced accuracy

    y_test (list or array): true labels
    y_pred (list or array): predicted labels.

    returns balanced accuracy score (float)
    """
    #checking if integers in list or strings
    if y_test[0] == 1 or y_test[0] == 0:
        num_healthy_true = np.sum(y_test)
    else:
        num_healthy_true = np.sum([x=='Healthy' for x in y_test])

    num_unhealthy_true = len(y_test) - num_healthy_true

    count_healthy_accurate = 0
    count_unhealthy_accurate = 0
    for i in range(0, len(y_test)):
        if (y_pred[i] == y_test[i] == 'Unhealthy') or (y_pred[i] == y_test[i] == 0):
            count_unhealthy_accurate +=1
        elif (y_pred[i] == y_test[i] == 'Healthy') or (y_pred[i] == y_test[i] == 1):
            count_healthy_accurate +=1

    healthy_percentage = count_healthy_accurate/num_healthy_true
    unhealthy_percentage = count_unhealthy_accurate/num_unhealthy_true
    balanced_accuracy = (healthy_percentage + unhealthy_percentage) * 0.5
    return balanced_accuracy

def get_specificity(y_test, y_pred):
    """
    aka true negative rate

    y_test (list or array): true labels
    y_pred (list or array): predicted labels.

    returns specificty score (float)
    """
    true_negative = 0
    false_positive = 0
    for i in range(0, len(y_test)):
        if (y_pred[i] == y_test[i] == 'Unhealthy') or (y_pred[i] == y_test[i] == 0):
            true_negative += 1
        elif y_pred[i] != y_test[i] and ((y_test[i] == 'Unhealthy') or (y_test[i] == 0)):
            false_positive += 1
    
    return true_negative / (true_negative+false_positive)

def get_precision(y_test, y_pred):
    """
    precision of positive (healthy) predictions
    """
    true_positive = 0
    false_positive = 0
    for i in range(0, len(y_test)):
        if (y_pred[i] == y_test[i] == 'Healthy') or (y_pred[i] == y_test[i] == 1):
            true_positive += 1
        elif y_pred[i] != y_test[i] and ((y_test[i] == 'Unhealthy') or (y_test[i] == 0)):
            false_positive += 1
    return true_positive/(true_positive + false_positive)

def get_negative_precision(y_test, y_pred):
    """
    precision of negative (unhealthy) predictions
    """
    true_positive = 0
    false_positive = 0
    for i in range(0, len(y_test)):
        if (y_pred[i] == y_test[i] == 'Unhealthy') or (y_pred[i] == y_test[i] == 0):
            true_positive += 1
        elif y_pred[i] != y_test[i] and ((y_test[i] == 'Healthy') or (y_test[i] == 1)):
            false_positive += 1
    return true_positive/(true_positive + false_positive)

def get_recall(y_test, y_pred):
    false_negative = 0
    true_positive = 0
    for i in range(0, len(y_test)):
        if (y_pred[i] == y_test[i] == 'Healthy') or (y_pred[i] == y_test[i] == 1):
            true_positive += 1
        elif y_pred[i] != y_test[i] and ((y_test[i] == 'Healthy') or (y_test[i] == 1)):
            false_negative += 1
    return true_positive/(true_positive + false_negative)

def get_negative_recall(y_test, y_pred):
    false_negative = 0
    true_positive = 0
    for i in range(0, len(y_test)):
        if (y_pred[i] == y_test[i] == 'Unhealthy') or (y_pred[i] == y_test[i] == 0):
            true_positive += 1
        elif y_pred[i] != y_test[i] and ((y_test[i] == 'Unhealthy') or (y_test[i] == 0)):
            false_negative += 1
    return true_positive/(true_positive + false_negative)


def get_f1_score(y_test, y_pred):
    """
    balance between precision and recall
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for i in range(0, len(y_test)):
        if (y_pred[i] == y_test[i] == 'Healthy') or (y_pred[i] == y_test[i] == 1):
            true_positive += 1
        elif y_pred[i] != y_test[i] and ((y_test[i] == 'Unhealthy') or (y_test[i] == 0)) :
            false_positive += 1
        elif y_pred[i] != y_test[i] and ((y_test[i] == 'Healthy') or (y_test[i] == 1)):
            false_negative += 1
    return (2*true_positive)/(2*true_positive+false_positive+false_negative)

def get_negative_f1_score(y_test, y_pred):
    """
    balance between precision and recall for negative (unhealthy) predictions
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for i in range(0, len(y_test)):
        if (y_pred[i] == y_test[i] == 'Unhealthy') or (y_pred[i] == y_test[i] == 0):
            true_positive += 1
        elif y_pred[i] != y_test[i] and ((y_test[i] == 'Healthy') or (y_test[i] == 1)) :
            false_positive += 1
        elif y_pred[i] != y_test[i] and ((y_test[i] == 'Unhealthy') or (y_test[i] == 0)):
            false_negative += 1
    return (2*true_positive)/(2*true_positive+false_positive+false_negative)

def scoring_function(model, X, y):
    """
    can be used to create a custom scoring function

    model: classification model to be used to make predictions
    X: data
    y: labels

    return float representing performance
    """
    
    y_pred = model.predict(X)
    y_test = y
    balanced_acc = get_balanced_accuracy(y_test, y_pred)

    f1 = get_f1_score(y_test, y_pred)
    
    
    return f1*0.7 + balanced_acc*0.3

def objective_score(y_test, y_pred):
    """
    can be used to create custom scoring function
    """
    balanced_acc = get_balanced_accuracy(y_test, y_pred)

    f1 = get_f1_score(y_test, y_pred)
    
    
    return f1*0.7 + balanced_acc*0.3

def get_av_confusion_matrix(y_test, y_pred):
    """
    function to average several confusion matrices from each skfold iteration
    """
    av_confusion_mat = np.zeros(shape = (len(y_test), 2, 2))
    for i in range(0, len(y_test)):
        av_confusion_mat[i] = confusion_matrix(y_test[i], y_pred[i])
    return np.mean(av_confusion_mat, axis=0)

def print_scores_for_channel(all_scores):
    """
    prints score metrics for each channel provided inputted dictionary of scores

    """

    # get channel names from keys
    channels = all_scores.keys()
    score_metrics = list(next(iter(all_scores.values())).keys())

    # dynamically construct dictionary
    data = {'Success Metric': score_metrics}

    for channel in channels:
        channel_key = f'Channel {channel}'
        channel_scores = [f"{all_scores[channel][metric]}" for metric in score_metrics]
        data[channel_key] = channel_scores

    # printing pandas dataframe
    df = pd.DataFrame(data)
    print(df)

    return None


def get_all_metrics(y_test, y_pred):
    """
    function to claculate all score metrics (postitve (healthy) predictions)
    returns dictionary of each metric
    """
    metrics = {}
    metrics['bal acc'] = get_balanced_accuracy(y_test, y_pred)
    metrics['accuracy'] = get_accuracy(y_test, y_pred)
    metrics['f1'] = get_f1_score(y_test, y_pred)
    metrics['recall'] = get_recall(y_test, y_pred)
    metrics['precision'] = get_precision(y_test, y_pred)

    return metrics

def get_all_weighted_averaged_metrics(y_test, y_pred, class_weight):
    """
    calculates the weighted metrics (postive and negative accuracy)
    returns dictionary of metrics
    """
    metrics = {}
    metrics['bal acc'] = get_balanced_accuracy(y_test, y_pred)
    metrics['accuracy'] = get_accuracy(y_test, y_pred)

    positive_f1_score = get_f1_score(y_test, y_pred)
    positive_recall = get_recall(y_test, y_pred)
    positive_precision = get_precision(y_test, y_pred)

    negative_f1_score = get_negative_f1_score(y_test, y_pred)
    negative_recall = get_negative_recall(y_test, y_pred)
    negative_precision = get_negative_precision(y_test, y_pred)

    weighted_f1_score = positive_f1_score * class_weight[1] + negative_f1_score * class_weight[0]
    weighted_recall = positive_recall * class_weight[1] + negative_recall * class_weight[0]
    weighted_precision = positive_precision * class_weight[1] + negative_precision * class_weight[0]

    metrics['recall'] = weighted_recall
    metrics['precision'] = weighted_precision
    metrics['f1'] = weighted_f1_score

    return metrics

def manual_y_predict(average_probs, threshold):
    """
    function to obtain manual preditions from probability arrays based on the thresholds for that probaility array

    returns array of maunal predictions
    """
    manual_y_pred = np.empty(len(average_probs), dtype=object)
    for j in range(0, len(average_probs)):
        if average_probs[j] > threshold:
            manual_y_pred[j] = 'Healthy'
        elif average_probs[j] <= threshold:
            manual_y_pred[j] = 'Unhealthy'
        elif average_probs[j] is None:
            manual_y_pred[j] = np.nan
    return manual_y_pred