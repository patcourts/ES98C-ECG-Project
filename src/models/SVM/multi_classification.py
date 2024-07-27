from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix, balanced_accuracy_score

import numpy as np

def ovo_skfold(params, health_state, n_splits, best_estimator, scorer=balanced_accuracy_score):
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) #can do repeated skf for better validation
    
    y_tests = []
    y_preds = []

    for train_index, test_index in skf.split(params, health_state):
        #getting test and train data sets
        X_train, X_test = params[train_index], params[test_index]
        y_train, y_test = health_state[train_index], health_state[test_index]
        
        #fitting data on previously calculated best estimator
        best_estimator.fit(X_train, y_train)
        
        #get prediction
        y_pred = best_estimator.predict(X_test)
        
        #return for calculation later
        y_preds.append(y_pred)
        y_tests.append(y_test)
        
    return y_tests, y_preds

#generate confusion matrix
def get_av_confusion_matrix(y_test, y_pred):
    av_confusion_mat = np.zeros(shape = (len(y_test), 5, 5))
    for i in range(0, len(y_test)):
        av_confusion_mat[i] = confusion_matrix(y_test[i], y_pred[i])
    return np.mean(av_confusion_mat, axis=0)

