from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import numpy as np
from models.scoring_metrics import get_all_weighted_averaged_metrics, get_f1_score, get_all_metrics, manual_y_predict
import itertools

def tune_hyperparams(params, labels, param_grid, classifier, scorer='balanced_accuracy'):
    """
    fucntion that perfroms a gridsearch to find the best hyperparameters for the classifier based on the parameters

    params: array of parameters to be classified
    labels: corresponding labels 
    param_grid: dictionary containing the hyperparameters for the gridsearch to iterate over
    classifier: model with which the gridsearch is using to test the performane
    scorer: metric with which the performance is tested

    returns classifier with optimal parameters
    """
    
    X_train, X_test, y_train, y_test = train_test_split(params, labels, test_size=0.3, stratify=labels)

    # perform grid search
    grid_search = GridSearchCV(classifier, param_grid, cv=3, scoring=scorer)
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_


def perform_skfold(params, health_state, n_splits, best_estimator, get_probabilities = False):
    """
    function that splits data through SKfold then trains and evaluates model on each split

    params: selected features
    health_state: labels 
    n_splits: number of splits for skfold
    best_estimator: optimised classifier to be used
    get_probabilities: option to return probabilities for reconstruction

    returns average scores over each split or if get_probabilities is true the information needed to reconstruct probabilities
    """
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) #can do repeated skf for better validation
    
    probabilities = []
    sample_percentages = []
    y_tests = []
    accuracy = []
    bal_acc = []
    precision = []
    recall = []
    f1 = []
    test_indices = []

    for train_index, test_index in skf.split(params, health_state):
        
        #getting test and train data sets
        X_train, X_test = params[train_index], params[test_index]
        y_train, y_test = health_state[train_index], health_state[test_index]
        
        

        #fitting data on previously calculated best estimator
        best_estimator.fit(X_train, y_train)
        
        #calculating percentage of healthy in training data
        threshold = sum(y_train)/len(y_train)

        #manual predictions with probabilities
        y_probs = best_estimator.predict_proba(X_test)
        y_pred = (y_probs[:, 1] > threshold).astype(int)

        class_weights = [(1-threshold), threshold]
        
        #score_metrics = get_all_weighted_averaged_metrics(y_test, y_pred, class_weights)
        score_metrics = get_all_metrics(y_test, y_pred)

        f1.append(score_metrics['f1'])
        recall.append(score_metrics['recall'])
        accuracy.append(score_metrics['accuracy'])
        bal_acc.append(score_metrics['bal acc'])
        precision.append(score_metrics['precision'])
        
        #for evaluation of model later
        sample_percentages.append(threshold) 
        probabilities.append(y_probs)
        y_tests.append(y_test)
        
        #for reconstruction of full patient data
        test_indices.append(test_index)

    #creating score metric dictionary
    all_average_scores = {}
    all_average_scores['F1 score'] = np.mean(np.array(f1))
    all_average_scores['Bal Acc'] = np.mean(np.array(bal_acc))
    all_average_scores['Accuracy'] = np.mean(np.array(accuracy))
    all_average_scores['precision'] = np.mean(np.array(precision))
    all_average_scores['recall'] = np.mean(np.array(recall))

    if get_probabilities:

        return all_average_scores, sample_percentages, probabilities, y_tests, test_indices
    else:
        return all_average_scores


def average_probabilities(probs, channels):
    """
    function to calculate the average probability for desired channels
    """
    average_probs = []
    for channel in channels:
        average_probs.append(probs[channel])

    return np.nanmean(average_probs, axis=0)

def optimise_score_over_channels(reconstructed_probs, threshold, health_state):
    """
    function to find the channels that optimise the perfromance through exhaustive search

    reconstructed_probs: total average probabilities for each channel
    threshold: threshold for making prediction based on class weights
    health_state: true class labels

    returns optimised score and the indices for the channels that lead to this
    """
    #calculate all possible combinations of channel indices
    channel_indices_list = [0, 1, 2, 3, 4, 5]
    all_combinations = []
    for r in range(1, len(channel_indices_list) + 1):
        combination = list(itertools.combinations(channel_indices_list, r))
        all_combinations.extend(combination)

    best_score = 0
    best_channel_indices = []
    for combo in all_combinations:
        
        # Use combo to get channel indices
        selected_channel_indices = [channel_indices_list[i] for i in combo]
    
        
        combo_probs = average_probabilities(reconstructed_probs, selected_channel_indices)
        
        manual_pred = manual_y_predict(combo_probs, threshold)

        #dont have any channels, is all combined into one
        score = get_f1_score(health_state, manual_pred)
            
        if score > best_score:
            best_score = score
            best_channel_indices = selected_channel_indices
    return best_score, best_channel_indices