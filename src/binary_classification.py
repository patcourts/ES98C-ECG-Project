from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
import numpy as np
from scoring_metrics import get_balanced_accuracy

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