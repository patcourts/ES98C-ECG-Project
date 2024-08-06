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
    num_healthy_true = np.sum([x=='Healthy' for x in y_test])
    num_unhealthy_true = len(y_test) - num_healthy_true
    count_healthy_accurate = 0
    count_unhealthy_accurate = 0
    for i in range(0, len(y_test)):
        if y_pred[i] == y_test[i] == 'Unhealthy':
            count_unhealthy_accurate +=1
        elif y_pred[i] == y_test[i] == 'Healthy':
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
        if y_pred[i] == y_test[i] == 'Healthy':
            true_negative += 1
        elif y_pred[i] != y_test[i] and y_test[i] == 'Healthy':
            false_positive += 1
    
    return true_negative / (true_negative+false_positive)

def get_precision(y_test, y_pred):
    true_positive = 0
    false_positive = 0
    for i in range(0, len(y_test)):
        if y_pred[i] == y_test[i] == 'Unhealthy':
            true_positive += 1
        elif y_pred[i] != y_test[i] and y_test[i] == 'Healthy':
            false_positive += 1
    return true_positive/(true_positive + false_positive)

def get_recall(y_test, y_pred):
    false_negative = 0
    true_positive = 0
    for i in range(0, len(y_test)):
        if y_pred[i] == y_test[i] == 'Unhealthy':
            true_positive += 1
        elif y_pred[i] != y_test[i] and y_test[i] == 'Unhealthy':
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
        if y_pred[i] == y_test[i] == 'Unhealthy':
            true_positive += 1
        elif y_pred[i] != y_test[i] and y_test[i] == 'Healthy':
            false_positive += 1
        elif y_pred[i] != y_test[i] and y_test[i] == 'Unhealthy':
            false_negative += 1
    return (2*true_positive)/(2*true_positive+false_positive+false_negative)

def scoring_function(model, X, y):
    """
    change to incorporate balanced accuracy 
    """
    
    y_pred = model.predict(X)
    y_test = y
    balanced_acc = get_balanced_accuracy(y_test, y_pred)
    #specificity = get_specificity(y_test, y_pred)
    f1 = get_f1_score(y_test, y_pred)
    
    
    return f1*0.7 + balanced_acc*0.3

def objective_score(y_test, y_pred):
    """
    change to incorporate balanced accuracy 
    """
    balanced_acc = get_balanced_accuracy(y_test, y_pred)
    specificity = get_specificity(y_test, y_pred)
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

    # #presenting results as pandas df
    # data = {
    #     'Success Metric': ['Objective Score', 'Balanced Accuracy'],
    #     'Channel 1': [f'{scores[0]['objective score']}', f'{scores[0]['balanced_accuracy']}],
    #     'Channel 2': [f'{score_accuracy[1]}', f'{balanced_accuracy[1]}'],
    #     'Channel 3': [f'{score_accuracy[2]}', f'{balanced_accuracy[2]}'],
    #     'Channel 4': [f'{score_accuracy[3]}', f'{balanced_accuracy[3]}'],
    #     'Channel 5': [f'{score_accuracy[4]}', f'{balanced_accuracy[4]}'],
    #     'Channel 6': [f'{score_accuracy[5]}', f'{balanced_accuracy[5]}'],
        
    # }

    # df = pd.DataFrame(data)
    # print(df)

    # return None

