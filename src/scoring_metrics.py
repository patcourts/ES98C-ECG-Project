import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd


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

def print_scores_for_channel(score_accuracy, balanced_accuracy):

    #presenting results as pandas df
    data = {
        'Success Metric': ['Objective Score', 'Balanced Accuracy'],
        'Channel 1': [f'{score_accuracy[0]}', f'{balanced_accuracy[0]}'],
        'Channel 2': [f'{score_accuracy[1]}', f'{balanced_accuracy[1]}'],
        'Channel 3': [f'{score_accuracy[2]}', f'{balanced_accuracy[2]}'],
        'Channel 4': [f'{score_accuracy[3]}', f'{balanced_accuracy[3]}'],
        'Channel 5': [f'{score_accuracy[4]}', f'{balanced_accuracy[4]}'],
        'Channel 6': [f'{score_accuracy[5]}', f'{balanced_accuracy[5]}'],
        
    }

    df = pd.DataFrame(data)
    print(df)
