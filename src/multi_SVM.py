from database.data import Data
from models.SVM.binary_classification import get_best_estimators
from models.SVM.multi_classification import ovo_skfold, get_av_confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC
from models.scoring_metrics import scoring_function, print_scores_for_channel
import numpy as np


#creating DATA object
ptb_multi_SVM = Data(database = 'ptbdb', denoise_method='DWT', train_splits=None, binary = False, parameterisation = True, estimation_method='SVM')

ptb_multi_SVM.run()

# define hyperparameter grid to test
param_grid = {
    'estimator__C': [0.01, 0.1, 1, 10],
    'estimator__kernel': ['linear', 'rbf', 'poly'],
    'estimator__gamma': ['scale']#including 'auto' aswell takes forever
}

#define multi classifier
ovo_svm = OneVsOneClassifier(SVC(class_weight = 'balanced', probability=True))


#find the best set of hyperparameters for each channel, tuned on the desired scoring function
best_estimators = get_best_estimators(ptb_multi_SVM.input_data, param_grid, 'balanced_accuracy', ovo_svm, ptb_multi_SVM.health_state, ptb_multi_SVM.nan_indices, labels = ptb_multi_SVM.labels)


#perform 3 way skfold to test accuracy
n_splits = 3

av_balanced_accuracy = []
confusion_mat = []
for i in range(0, 6):
    y_test, y_pred = ovo_skfold(ptb_multi_SVM.input_data[i], ptb_multi_SVM.labels[i], n_splits, best_estimators[i])
    balanced_accuracy = []
    for j in range(n_splits):
        balanced_accuracy.append(balanced_accuracy_score(y_test[j], y_pred[j]))

    confusion_mat.append(get_av_confusion_matrix(y_test, y_pred))
    av_balanced_accuracy.append(np.mean(balanced_accuracy))

#printing scores for each channel
print_scores_for_channel(av_balanced_accuracy, av_balanced_accuracy)#fix this later
