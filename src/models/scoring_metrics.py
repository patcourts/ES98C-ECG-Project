import numpy as np
from sklearn.metrics import confusion_matrix

import pandas as pd

def get_accuracy(y_test, y_pred):
    """
    Calculate the accuracy.

    Parameters:
    y_test (list or array): True labels.
    y_pred (list or array): Predicted labels.

    Returns:
    float: Accuracy score.
    """
    # count the number of correct predictions (TP + TN)
    correct_predictions = np.sum(np.array(y_test) == np.array(y_pred))
    
    # divide by total datapoints (TP + TN + FN + FP)
    accuracy = correct_predictions / len(y_test)
    
    return accuracy


def get_balanced_accuracy(y_test, y_pred):
    """
    balanced accuracy....
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
    true negative rate
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
    true_positive = 0
    false_positive = 0
    for i in range(0, len(y_test)):
        if (y_pred[i] == y_test[i] == 'Healthy') or (y_pred[i] == y_test[i] == 1):
            true_positive += 1
        elif y_pred[i] != y_test[i] and ((y_test[i] == 'Unhealthy') or (y_test[i] == 0)):
            false_positive += 1
    return true_positive/(true_positive + false_positive)

def get_negative_precision(y_test, y_pred):
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
    balance between precision and recall
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
    change to incorporate balanced accuracy 
    """
    
    y_pred = model.predict(X)
    y_test = y
    balanced_acc = get_balanced_accuracy(y_test, y_pred)

    f1 = get_f1_score(y_test, y_pred)
    
    
    return f1*0.7 + balanced_acc*0.3

def objective_score(y_test, y_pred):
    """
    change to incorporate balanced accuracy 
    """
    balanced_acc = get_balanced_accuracy(y_test, y_pred)

    f1 = get_f1_score(y_test, y_pred)
    
    
    return f1*0.7 + balanced_acc*0.3

def get_av_confusion_matrix(y_test, y_pred):
    av_confusion_mat = np.zeros(shape = (len(y_test), 2, 2))
    for i in range(0, len(y_test)):
        av_confusion_mat[i] = confusion_matrix(y_test[i], y_pred[i])
    return np.mean(av_confusion_mat, axis=0)

def print_scores_for_channel(all_scores):
    """
    Prints score metrics for each channel.
    
    Parameters:
    all_scores (dict): A dictionary where the key is the channel number and the value is another dictionary
                       containing various score metrics.
    """

    # Extract the score metrics from the first channel
    channels = all_scores.keys()
    score_metrics = list(next(iter(all_scores.values())).keys())

    # Constructing the data dictionary dynamically
    data = {'Success Metric': score_metrics}

    for channel in channels:
        channel_key = f'Channel {channel}'
        channel_scores = [f"{all_scores[channel][metric]}" for metric in score_metrics]
        data[channel_key] = channel_scores

    # Creating and displaying the DataFrame
    df = pd.DataFrame(data)
    print(df)

    return None


def get_all_metrics(y_test, y_pred):
    metrics = {}
    metrics['bal acc'] = get_balanced_accuracy(y_test, y_pred)
    metrics['accuracy'] = get_accuracy(y_test, y_pred)
    metrics['f1'] = get_f1_score(y_test, y_pred)
    metrics['recall'] = get_recall(y_test, y_pred)
    metrics['precision'] = get_precision(y_test, y_pred)

    return metrics

def get_all_weighted_averaged_metrics(y_test, y_pred, class_weight):
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

