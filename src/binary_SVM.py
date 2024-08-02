from database.utils import get_reconstructed_probabilities
from models.SVM.binary_classification import get_best_estimators, get_scores_and_probs, optimise_score_over_channels
from models.scoring_metrics import scoring_function, print_scores_for_channel
from database.data import Data
from sklearn.svm import SVC
from tqdm import tqdm

#creating DATA object
ptb_binary_SVM = Data(database = 'ptbdb', denoise_method='DWT', train_splits=None, binary = True, parameterisation = True)



# define hyperparameter grid to test
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale']#including 'auto' aswell takes forever
}


ptb_binary_SVM.run()

#define classifier
svc = SVC(class_weight='balanced', probability = True)

#find the best set of hyperparameters for each channel, tuned on the desired scoring function
best_estimators = get_best_estimators(ptb_binary_SVM.selected_features, param_grid, scoring_function, svc, ptb_binary_SVM.health_state, ptb_binary_SVM.nan_indices)

#perform 3 way skfold to get scores for each channel as well as their probabilities
n_splits = 3
score, bal_acc, probabilities, thresholds, y_tests, test_indices = get_scores_and_probs(ptb_binary_SVM.selected_features, ptb_binary_SVM.health_state, ptb_binary_SVM.nan_indices, best_estimators, scoring_function, n_splits)

#printing scores for each channel
print_scores_for_channel(score, bal_acc)

#reconstructing calculated probabilities so can optimise over all channels
reconstructed_probs = get_reconstructed_probabilities(probabilities, test_indices, ptb_binary_SVM.nan_indices, ptb_binary_SVM.allowed_patients.count_patients(), n_splits)

#optimising over channels
best_score, best_channel_indices = optimise_score_over_channels(reconstructed_probs, thresholds, ptb_binary_SVM.health_state)

print(best_score, best_channel_indices)



