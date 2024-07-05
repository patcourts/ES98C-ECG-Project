from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
import numpy as np
from scoring_metrics import get_balanced_accuracy, objective_score
import itertools

def tune_hyperparams(params, health_state, param_grid, scorer='balanced_accuracy'):
    
    X_train, X_test, y_train, y_test = train_test_split(params, health_state, test_size=0.3, stratify=health_state)
    
    #initialise classifier
    svc = SVC(class_weight='balanced', probability = True)

    # perform grid search
    grid_search = GridSearchCV(svc, param_grid, cv=3, scoring=scorer)
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

def get_best_estimators(params, param_grid, scoring_function, health_state, nan_indices):
    best_estimators = []
    for i in range(0, len(nan_indices)):
        best_estimators.append(tune_hyperparams(params[i], health_state[nan_indices[i]], param_grid, scoring_function))
    return best_estimators

def skfold_with_probabilities(params, health_state, n_splits, best_estimator, scorer):
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) #can do repeated skf for better validation
    
    probabilities = []
    sample_percentages = []
    y_tests = []
    balanced_accuracy = []
    score_func = []
    test_indices = []
    for train_index, test_index in skf.split(params, health_state):
        #getting test and train data sets
        X_train, X_test = params[train_index], params[test_index]
        y_train, y_test = health_state[train_index], health_state[test_index]
        
        
        #calculating percentage of healthy in training data
        sample_percentages.append(np.mean(y_train == 'Healthy')) #how is threshold measured??? health_state[train_index]??
        
        #fitting data on previously calculated best estimator
        best_estimator.fit(X_train, y_train)
        
        #evaluating model
        y_pred = best_estimator.predict(X_test)
        score_func.append(scorer(best_estimator, X_train, y_test))
        balanced_accuracy.append(get_balanced_accuracy(y_test, y_pred))
        
        #for evaluation of model later
        probabilities.append(best_estimator.predict_proba(X_test))
        y_tests.append(y_test)
        
        #for reconstruction of full patient data
        test_indices.append(test_index)

    return np.mean(np.array(score_func)), np.mean(np.array(balanced_accuracy)), probabilities, sample_percentages, y_tests, test_indices

def get_scores_and_probs(params, health_state, nan_indices, best_estimators, scoring_function, n_splits=3):
    score_accuracy = []
    balanced_accuracy = []
    n_splits=n_splits

    probs = []
    thresholds = []
    y_tests = []
    test_indices = []

    for i in range(0, len(nan_indices)):
        score_acc, bal_acc, prob, threshold, y_test, test_indice = skfold_with_probabilities(params[i], health_state[nan_indices[i]], n_splits, best_estimators[i], scoring_function)
        score_accuracy.append(score_acc)
        balanced_accuracy.append(bal_acc)
        probs.append(prob)
        thresholds.append(threshold)
        y_tests.append(y_test)
        test_indices.append(test_indice)
    
    return score_accuracy, balanced_accuracy, probs, thresholds, y_tests, test_indices

def average_probabilities(probs, channels):
    average_probs = []
    for channel in channels:
        average_probs.append(probs[channel])

    return np.nanmean(average_probs, axis=0)

def manual_y_predict(average_probs, threshold):
    manual_y_pred = np.empty(len(average_probs), dtype=object)
    for j in range(0, len(average_probs)):
        if average_probs[j] > 0.3:
            manual_y_pred[j] = 'Healthy'
        elif average_probs[j] <= 0.3:
            manual_y_pred[j] = 'Unhealthy'
        elif average_probs[j] is None:
            manual_y_pred[j] = np.nan
    return manual_y_pred

def optimise_score_over_channels(reconstructed_probs, thresholds, health_state):
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
        
        #print(selected_channel_indices)
        #no longer have any splits
        #combo_probs = average_probabilities(probs, n_splits, selected_channel_indices)
        #manual_pred = manual_y_pred(combo_probs, n_splits, thresholds)
        
        combo_probs = average_probabilities(reconstructed_probs, selected_channel_indices)
        manual_pred = manual_y_predict(combo_probs, thresholds)


        #dont have any channels, is all combined into one
        score = objective_score(health_state, manual_pred)
            
        if score > best_score:
            best_score = score
            best_channel_indices = selected_channel_indices
    return best_score, best_channel_indices